# File for compressing some uncompressed .npz files.
# Original code was created via ChatGPT, then was edited with a few customizations.
import numpy as np
import os

def compress_npz(input_dir, output_postfix = "_c"):
    """
    Compresses all uncompressed .npz files in the input directory,
    saves them to the output directory, and verifies the data.
    """

    for file in os.listdir(input_dir):
        f_split = os.path.splitext(file)
        if file.endswith(".npz") and not f_split[0].endswith(output_postfix):
            out_name = f_split[0] + output_postfix + f_split[1]

            output_path = os.path.join(input_dir, out_name)
            input_path = os.path.join(input_dir, file)

            # Load the uncompressed .npz file
            data = np.load(input_path)

            # Save it in compressed format
            compressed_data = {key: data[key] for key in data.files}
            np.savez_compressed(output_path, **compressed_data)

            # Verify that the compressed file matches the original
            with np.load(output_path) as new_data:
                for key in data.files:
                    original = data[key]
                    compressed = new_data[key]
                    if not np.array_equal(original, compressed, True):
                        diff_inds = original != compressed
                        diff_orig = original[diff_inds]
                        diff_compressed = compressed[diff_inds]
                        raise ValueError(
                            f"Data mismatch for key '{key}' in file {file}: \
                                {diff_orig} vs. {diff_compressed}"
                            )
            print(f"Compressed and verified: {file}")

# Specify the input and output directories
input_directory = "."

compress_npz(input_directory)
