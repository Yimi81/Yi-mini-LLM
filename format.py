def best_fit_decreasing(strings, max_seq_len):
    # Step 1: Sort strings by length in decreasing order
    strings.sort(key=len, reverse=True)
    
    # Step 2: Prepare to store the result
    result = []
    used = [False] * len(strings)
    
    # Step 3: Try to fit smaller strings into the space remaining in larger strings
    for i in range(len(strings)):
        if not used[i]:
            # Current string as the base
            current = strings[i]
            used[i] = True
            # Try to append other strings to it
            for j in range(i + 1, len(strings)):
                if not used[j] and len(current) + len(strings[j]) <= max_seq_len:
                    current += strings[j]
                    used[j] = True
            # Append the combined string to the result list
            result.append(current)
    
    return result

text_examples = ["very long text that to be split", "another example", "short", "hello world, this is ygf"]
max_seq_length = 10

# 将每个字符串按照最大长度max_seq_length分割
split_texts = [text[i:i+max_seq_length] for text in text_examples for i in range(0, len(text), max_seq_length)]
split_texts_len = [len(text) for text in split_texts]
remain_capacity = [max_seq_length - len(text) for text in split_texts]

sorted_split_texts = sorted(split_texts, key=lambda x: sum(len(part) for part in x), reverse=True)
print(split_texts)
print(split_texts_len)
print(remain_capacity)

best_fit_pack = best_fit_decreasing(split_texts, max_seq_length)
result = {idx: value for idx, value in enumerate(best_fit_pack)}
print(result)
