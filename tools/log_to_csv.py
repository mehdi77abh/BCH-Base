import os
import csv
import numpy as np

def log_to_csv(csv_path, ber, tmr_per, iteration, acc, tacc, prec, rec,
               conf_matrix, sub_conf, acc_50):
    """
    Append one experiment row to the given CSV file.
    conf_matrix and sub_conf are stored as JSON strings.
    """
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline='') as f:
        writer = csv.writer(f)

        # Write header if file is empty
        if not file_exists or os.path.getsize(csv_path) == 0:
            writer.writerow([
                "BER", "TMR_percent", "Iteration",
                "Accuracy", "Top5_Accuracy", "Precision", "Recall",
                "Confusion_Matrix", "Sub_Confusion_Matrix", "Acc_50"
            ])

        # Convert matrices to JSON strings for compact storage
        import json
        conf_json = json.dumps(conf_matrix)
        sub_conf_json = json.dumps(sub_conf)

        writer.writerow([
            f"{ber:.0e}",    # scientific notation
            tmr_per,
            iteration,
            f"{acc:.4f}",
            f"{tacc:.4f}",
            f"{prec:.4f}",
            f"{rec:.4f}",
            conf_json,
            sub_conf_json,
            f"{acc_50:.4f}"
        ])
        f.flush()
        os.fsync(f.fileno())