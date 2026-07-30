import matplotlib.pyplot as plt

def plot_logs(log_file="logs.txt"):
    epochs = []
    train_accs = []
    test_accs = []

    try:
        with open(log_file, "r") as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            # we are looking for lines like: "epoch 1 - train acc: 0.55, test acc: 0.60"
            if line.startswith("epoch"):
                parts = line.split("-")
                if len(parts) >= 2:
                    epoch_part = parts[0].strip()  # "epoch 1"
                    metrics_part = parts[1].strip() # "train acc: 0.55, test acc: 0.60"
                    
                    # extract numbers
                    epoch_num = int(epoch_part.split()[1])
                    
                    metrics = metrics_part.split(",")
                    train_acc = float(metrics[0].split(":")[1].strip())
                    test_acc = float(metrics[1].split(":")[1].strip())
                    
                    epochs.append(epoch_num)
                    train_accs.append(train_acc)
                    test_accs.append(test_acc)
                    
        if not epochs:
            print("No valid epoch data found in logs.txt. Make sure you saved the file!")
            return

        # Create the plot
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_accs, marker='o', label='Training Accuracy', color='blue')
        plt.plot(epochs, test_accs, marker='s', label='Testing Accuracy', color='orange')
        
        plt.title('Model Accuracy over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(epochs) # Show all epochs on x-axis
        
        # Save and show the plot
        plt.savefig("accuracy_graph.png")
        print("Graph saved as accuracy_graph.png")
        plt.show()

    except FileNotFoundError:
        print(f"Error: {log_file} not found. Please create it and paste your training logs.")

if __name__ == "__main__":
    plot_logs()
