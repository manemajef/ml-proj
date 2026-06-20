import pandas as pd

TEST_INPUT = "Test_Data_No_Target.csv"
TRAIN_INPUT = "Train_Data.csv"


def sample_csv(input, output, n=1000):
    df = pd.read_csv(input)
    sample = df.sample(n=n, random_state=42)
    sample.to_csv(output, index=False)
    print(f"Saved {output}.csv")


def main():
    sample_csv(TEST_INPUT, "test-sample.csv")
    sample_csv(TRAIN_INPUT, "train-sample.csv")


if __name__ == "__main__":
    main()
