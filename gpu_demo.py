import torch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    y = torch.tensor([4.0, 5.0, 6.0], device=device)
    print("使用设备:", device)
    print("计算结果:", x + y)

if __name__ == "__main__":
    main()