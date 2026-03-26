import pandas as pd
import re


def load_with_numbers(input_path, output_path=None):
    data = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            match = re.match(r"<(\d+):\d+(?::[^>]+)?>\s*(.*)", line)

            if match:
                doc_id = int(match.group(1))   # first number
                text = match.group(2)          # the text after >

                #data.append((doc_id, text))
                data.append(text)

    df = pd.DataFrame(data, columns=["text"])
    if output_path is None:
        return df

    df.to_parquet(output_path, index=False)


if __name__ == "__main__":
    load_with_numbers("Dataset/test/corpus.tache1.test.utf8", "Dataset/clean/presidents_test.parquet")
