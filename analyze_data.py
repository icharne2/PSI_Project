import matplotlib.pyplot as plt
import seaborn as sns

def analyze_data(train_data, test_data, save_path_prefix="analysis"):
    print("Train data shape:", train_data.shape)
    print("Test data shape:", test_data.shape)

    print("\nTrain data info:")
    train_data.info()
    print("\nTest data info:")
    test_data.info()

    print("\nChecking for missing values:")
    print("Train missing values:\n", train_data.isnull().sum().sum())
    print("Test missing values:\n", test_data.isnull().sum().sum())

    print("\nTrain label distribution:\n", train_data['label'].value_counts())
    print("Test label distribution:\n", test_data['label'].value_counts())

    print("\nBasic statistics:")
    print(train_data.describe())

    # Label range check
    print("\nLabel range check:")
    print("Train labels in range 0-9:", train_data['label'].between(0, 9).all())
    print("Test labels in range 0-9:", test_data['label'].between(0, 9).all())

    # Checking class balance
    print("\nTrain label variance:", train_data['label'].value_counts().std())
    print("Test label variance:", test_data['label'].value_counts().std())

    # Label layout visualization for train data
    plt.figure(figsize=(10, 5))
    ax = sns.countplot(x=train_data['label'], palette='viridis', hue=train_data['label'], legend=False,
                       order=sorted(train_data['label'].unique()))
    plt.title("Train Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Number of occurrences")

    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom', fontsize=10, color='black')

    if save_path_prefix:
        plt.savefig(f"{save_path_prefix}_train_distribution.png")
    plt.show()

    # Label layout visualization for test data
    plt.figure(figsize=(10, 5))
    ax = sns.countplot(x=test_data['label'], palette='viridis', hue=test_data['label'], legend=False,
                       order=sorted(test_data['label'].unique()))
    plt.title("Test Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Number of occurrences")

    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom', fontsize=10, color='black')

    if save_path_prefix:
        plt.savefig(f"{save_path_prefix}_test_distribution.png")
    plt.show()

    # Sample images
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    for i, ax in enumerate(axes):
        img = train_data.iloc[i, 1:].values.reshape(28, 28)
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Label: {train_data.iloc[i, 0]}")
        ax.axis("off")

    if save_path_prefix:
        plt.savefig(f"{save_path_prefix}_sample_images.png")
    plt.show()

    # Average images for each class
    plt.figure(figsize=(10, 5))
    for label in sorted(train_data['label'].unique()):
        mean_img = train_data[train_data['label'] == label].drop(columns=['label']).mean().values.reshape(28, 28)
        plt.subplot(2, 5, label + 1)
        plt.imshow(mean_img, cmap='gray')
        plt.title(f"Label {label}")
        plt.axis("off")

    plt.suptitle("Average images for each class")
    if save_path_prefix:
        plt.savefig(f"{save_path_prefix}_class_averages.png")
    plt.show()
