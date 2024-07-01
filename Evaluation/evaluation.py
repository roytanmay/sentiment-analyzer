from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def precision_score_plot(y_test, y_score):
    precision, recall, thresholds = precision_recall_curve(y_test, y_score)

    fig, ax = plt.subplots()
    ax.plot(recall, precision, color='purple')
    ax.set_title("Precision Recall Curve")
    ax.set_ylabel("Precision")
    ax.set_xlabel("Recall")

    plt.show()


def confusion_matrix_plot(y_test, y_score):
    confmatrix = confusion_matrix(y_test, y_score)
    fig, ax = plt.subplots(figsize=(3, 3))
    
    ax.imshow(confmatrix)
    ax.grid(False)
    ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
    ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
    ax.set_ylim(1.5, -0.5)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, confmatrix[i, j], ha='center', va='center', color='red')

    plt.show()