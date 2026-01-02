import argparse
from pathlib import Path
import re
from typing import List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

BASE_DIR = Path(__file__).parent
EMBEDDING_MODEL_PATH = BASE_DIR / "embedding-model"
DATA_PATH = BASE_DIR / "data" / "processed_true_data.txt"


def parse_curly_blocks(text: str) -> List[str]:
    blocks: List[str] = []
    buf: List[str] = []
    depth = 0

    for ch in text:
        if ch == "{":
            if depth == 0:
                buf = []
            depth += 1

        if depth > 0:
            buf.append(ch)

        if ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0:
                    block = "".join(buf).strip()
                    if block:
                        blocks.append(block)

    return blocks


def extract_index_key(block: str) -> Optional[str]:
    m = re.search(r"\[([^\]]+)\]", block)
    if not m:
        return None
    name = m.group(1).strip()
    return name if name else None


def extract_answer_and_explanation(block: str) -> Tuple[Optional[str], Optional[str]]:
    answer = None
    explanation = None

    m_ans = re.search(r"正确答案[:：]\s*([A-E])", block)
    if m_ans:
        answer = m_ans.group(1).strip()

    m_exp = re.search(r"答案解析[:：]\s*(.+)", block)
    if m_exp:
        explanation = m_exp.group(1).strip()

    return answer, explanation


def parse_gold_answers(test_path: Path) -> List[str]:
    """解析 test.txt：每行以选项字母开头，如 'E （解释）'。返回从第1题开始的答案列表。"""
    text = test_path.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    gold: List[str] = []
    pat = re.compile(r"^(?:(\d+)\s*[\.→\-:：]?\s*)?([A-E])\b")
    for ln in lines:
        m = pat.match(ln)
        if not m:
            continue
        gold.append(m.group(2))

    return gold


class RAGSystem:
    def __init__(self, verbose: bool = True):
        self.tokenizer = None
        self.model = None
        self.keys: List[str] = []
        self.blocks: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.verbose = verbose
        
    def load_model(self):
        """加载embedding模型"""
        if self.verbose:
            print(f"正在加载embedding模型: {EMBEDDING_MODEL_PATH}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(EMBEDDING_MODEL_PATH))
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(str(EMBEDDING_MODEL_PATH), torch_dtype=dtype)
        self.model.to(self.device)
        self.model.eval()
        if self.verbose:
            print(f"模型加载完成，使用设备: {self.device}")
        
    def load_documents(self, data_path: Path = DATA_PATH):
        """加载文档数据（按{}切割为知识块，并提取[]中的索引key）"""
        if self.verbose:
            print(f"正在加载文档: {data_path}")
        raw = data_path.read_text(encoding="utf-8")
        blocks = parse_curly_blocks(raw)

        keys: List[str] = []
        kept_blocks: List[str] = []
        for b in blocks:
            name = extract_index_key(b)
            if not name:
                continue
            keys.append(name)
            kept_blocks.append(b)

        self.keys = keys
        self.blocks = kept_blocks
        if self.verbose:
            print("")
        
    def get_embedding(self, text: str) -> np.ndarray:
        """获取文本的embedding向量"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :]
            embedding = embedding.float().cpu().numpy()
        return embedding[0]

    def get_embeddings(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """批量获取embedding向量"""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        all_embeddings: List[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                emb = outputs.last_hidden_state[:, 0, :]
                emb = emb.float().cpu().numpy()

            all_embeddings.append(emb)

        return np.concatenate(all_embeddings, axis=0)
    
    def build_index(self):
        """为所有文档构建embedding索引"""
        if self.verbose:
            print("正在构建文档索引...")
        embeddings = self.get_embeddings(self.keys, batch_size=16)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        self.embeddings = embeddings / norms
        if self.verbose:
            print("索引构建完成")
        
    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """搜索最相关的文档"""
        if self.embeddings is None or len(self.blocks) == 0:
            raise RuntimeError("索引尚未构建，请先调用 build_index()")

        query_emb = self.get_embedding(query)

        q_norm = np.linalg.norm(query_emb)
        q_norm = max(float(q_norm), 1e-12)
        query_emb = query_emb / q_norm

        similarities = np.dot(self.embeddings, query_emb)
        
        top_k = max(1, min(int(top_k), len(self.blocks)))
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(self.blocks[i], float(similarities[i])) for i in top_indices]
        return results


def main():
    parser = argparse.ArgumentParser(description="简单RAG检索演示（仅Embedding检索，不调用LLM）")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--show-score", action="store_true")
    parser.add_argument("--data", type=str, default=str(DATA_PATH))
    parser.add_argument("--output-mode", type=str, choices=["test", "block"], default="block")  # 默认改为block模式
    parser.add_argument("--eval-file", type=str, default=None)
    parser.add_argument("--eval-n", type=int, default=100)
    args = parser.parse_args()

    eval_mode = args.eval_file is not None

    rag_system = RAGSystem(verbose=not eval_mode)
    rag_system.load_model()
    rag_system.load_documents(Path(args.data))
    rag_system.build_index()

    if eval_mode:
        gold = parse_gold_answers(Path(args.eval_file))
        n = max(0, int(args.eval_n))
        n = min(n, len(gold), len(rag_system.keys))
        if n == 0:
            print("0")
            return

        correct = 0
        for i in range(n):
            q = rag_system.keys[i]
            doc, _ = rag_system.search(q, top_k=1)[0]
            pred, _ = extract_answer_and_explanation(doc)
            if pred == gold[i]:
                correct += 1

        acc = correct / n
        print(f"{acc:.6f}")
        return

    def run_query(q: str):
        results = rag_system.search(q, top_k=args.top_k)
        if not results:
            print("未检索到结果")
            return

        if args.output_mode == "block":
            # 直接打印Top-K知识块
            print(f"\n查询: {q}")
            print(f"返回Top-{args.top_k}相似知识块:\n")
            
            for idx, (doc, score) in enumerate(results, 1):
                if args.show_score:
                    print(f"--- 排名 {idx} (相似度: {score:.4f}) ---")
                else:
                    print(f"--- 排名 {idx} ---")
                print(doc)
                print()
            return

        # test模式（原答题逻辑，已移除）
        print("test模式已不再支持，请使用 --output-mode block")

    if args.query:
        run_query(args.query)
        return

    print(f"\n请输入问题进行检索（返回Top-{args.top_k}），输入 exit 退出。\n")
    while True:
        q = input("> ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            break
        run_query(q)


if __name__ == "__main__":
    main()