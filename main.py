from read_data import read_data
from transform_data import transform_data
from model import predict_and_save, train_rfc, evaluate_rfc
from sklearn.model_selection import train_test_split
from utils import SEED, plot_status_map

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

LABEL_MAP = {"functional": 0, "functional needs repair": 1, "non functional": 2}


def main():
    log.info("Reading and cleaning data...")
    train_df, train_labels_df, test_df = read_data()

    test_ids = test_df["id"]

    # plot_status_map(train_df, train_labels_df["status_group"])
    plot_status_map(train_df, train_labels_df["status_group"], "functional")
    plot_status_map(train_df, train_labels_df["status_group"], "functional needs repair")
    plot_status_map(train_df, train_labels_df["status_group"],"non functional")

    log.info("Transforming train data...")
    X, encoders = transform_data(train_df, fit_encoders=True)
    log.info("Transforming test data...")
    X_test, _ = transform_data(test_df, fit_encoders=False, encoders=encoders)
    y = train_labels_df["status_group"].map(LABEL_MAP)

    # stratify for class imbalance?
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y) 

    log.info("Training model...")
    rfc = train_rfc(X_train, y_train)
    log.info("Evaluating model...")
    evaluate_rfc(rfc, X_val, y_val)
    log.info("Predicting test data...")
    predict_and_save(rfc, X_test, test_ids)
    log.info("Done.")

if __name__ == "__main__":
    main()
