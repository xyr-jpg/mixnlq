import time
from itertools import islice

import torch
from datasets import load_dataset
from tqdm import tqdm

PPL_TASKS = ("wikitext2", "c4")


def _load_wikitext2_texts():
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    return dataset["text"]


def _load_c4_texts(max_samples=1000):
    if max_samples <= 0:
        raise ValueError("C4 max_samples must be positive")
    dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    return [record["text"] for record in islice(dataset, max_samples)]

#wikitext2
def load_wikitext_data(split="test", max_samples=None):
    """加载Wikitext数据集"""
    if split != "test":
        raise ValueError("WikiText-2 is loaded from the test split")
    print("加载Wikitext-2 test 数据集...")
    texts = _load_wikitext2_texts()
    if max_samples:
        texts = texts[:max_samples]
    
    # 过滤空行和无效文本
    filtered_texts = []
    for text in texts:
        cleaned_text = text.strip()
        if cleaned_text and len(cleaned_text) > 10:
            filtered_texts.append(cleaned_text)
    
    print(f"有效文本数量: {len(filtered_texts)}")
    return filtered_texts

def load_C4_en_data(split="validation", nsamples=1000):
    if split != "validation":
        raise ValueError("C4 is loaded from the validation split")
    texts = _load_c4_texts(nsamples)
    filtered_texts = []
    for text in texts:
        cleaned_text = text.strip()
        if cleaned_text and len(cleaned_text) > 10:
            filtered_texts.append(cleaned_text)
    
    print(f"有效文本数量: {len(filtered_texts)}")
    return filtered_texts

def calculate_perplexity(model, tokenizer, texts, batch_size=4, max_length=512, stride=1024, print_info=True):
    """计算困惑度"""
    encodings = tokenizer("\n\n".join(texts), return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    
    model.eval()
    nll_sum=0
    n_tokens=0
    prev_end_loc = 0
    time_cnt=0
    avg_time=0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        # try:
        with torch.no_grad():
            tick = time.time()
            outputs = model(input_ids, labels=target_ids)
            delta_time = time.time() - tick
            if time_cnt >= 1:
                avg_time *= ((time_cnt-1) / time_cnt)
                avg_time += (delta_time) / time_cnt
            time_cnt += 1
            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        # Accumulate the total negative log-likelihood and the total number of tokens
        num_valid_tokens = (target_ids != -100).sum().item()  # number of valid tokens in target_ids
        batch_size = target_ids.size(0)
        num_loss_tokens = num_valid_tokens - batch_size  # subtract batch_size due to internal label shift
        nll_sum += neg_log_likelihood * num_loss_tokens
        n_tokens += num_loss_tokens

        avg_nll = nll_sum / n_tokens  # average negative log-likelihood per token
        ppl = torch.exp(avg_nll)
        if print_info:
            print(f"\n困惑度 (Perplexity): {ppl:.2f}, 平均损失: {avg_nll:.4f}")
            print(f"{time_cnt-1} average time: {avg_time}")
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
        # except:
            # print("ppl error!!!")
            # pass
            
    avg_nll = nll_sum / n_tokens  # average negative log-likelihood per token
    ppl = torch.exp(avg_nll)
    return ppl, avg_nll


def evaluate_ppl_tasks(model, tokenizer, tasks, c4_max_samples):
    results = {}
    for task in tasks:
        if task == "wikitext2":
            texts = load_wikitext_data("test")
            split, stride = "test", 512
        elif task == "c4":
            texts = load_C4_en_data("validation", c4_max_samples)
            split, stride = "validation", 1024
        else:
            raise ValueError(f"unsupported PPL task: {task}")
        if not texts:
            raise RuntimeError(f"No usable text found for {task} ({split})")

        perplexity, average_loss = calculate_perplexity(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            batch_size=1,
            max_length=2048,
            stride=stride,
            print_info=False,
        )
        results[task] = {
            "split": split,
            "num_texts": len(texts),
            "max_length": 2048,
            "stride": stride,
            "perplexity": float(perplexity.item()),
            "average_loss": float(average_loss.item()),
        }
        print(
            f"{task} ({split}, {len(texts)} texts): "
            f"PPL={results[task]['perplexity']:.4f}, "
            f"average_loss={results[task]['average_loss']:.6f}"
        )
    return results


