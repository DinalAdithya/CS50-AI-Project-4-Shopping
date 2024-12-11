import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

TEST_SIZE = 0.4


def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )



    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")




def load_data(filename):
    evidence = []
    labels = []

    Month = {
        "Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3, "May": 4, "June": 5, "Jul": 6,
        "Aug": 7, "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11
    }

    with open(filename, mode='r') as file:
        csvFile = csv.reader(file)
        next(csvFile)  # to skip the header

        for row in csvFile:
            if not row:
                continue
            evidence.append([
                int(row[0]),  # Administrative
                float(row[1]),  # Administrative_Duration
                int(row[2]),  # Informational
                float(row[3]),  # Informational_Duration
                int(row[4]),  # ProductRelated
                float(row[5]),  # ProductRelated_Duration
                float(row[6]),  # BounceRates
                float(row[7]),  # ExitRates
                float(row[8]),  # PageValues
                float(row[9]),  # SpecialDay
                Month[row[10]],  # Month
                int(row[11]),  # OperatingSystems
                int(row[12]),  # Browser
                int(row[13]),  # Region
                int(row[14]),  # TrafficType
                1 if row[15] == "Returning_Visitor" else 0,  # VisitorType
                1 if row[16] == "TRUE" else 0,  # Weekend

            ])
            labels.append(1 if row[17].strip().lower() == "true" else 0)# Revenue

    return evidence, labels


def train_model(X_train, y_train):

    # Initialize the KNN model with k = 1
    model = KNeighborsClassifier(n_neighbors=1)

    # Train the model on training data set
    model.fit(X_train, y_train)

    return model


def evaluate(labels, predictions):

    TP = 0
    TN = 0
    FP = 0
    FN = 0


    for lab, pv in zip(labels, predictions):
        if lab == 1 and pv == 1:
            TP += 1
        elif lab == 1 and pv == 0:
            FN += 1
        elif lab == 0 and pv == 0:
            TN += 1
        else:
            FP += 1

    if TP+FN == 0:
        sensitivity = 0.0
    else:
        sensitivity = TP/(TP + FN) # this measures how well the model identify the positive cases

    if TN+FP == 0:
        specificity = 0.0
    else:
        specificity = TN/(TN + FP) # this measures how well the model identify the negative cases

    return (sensitivity, specificity)



if __name__ == "__main__":
    main()
