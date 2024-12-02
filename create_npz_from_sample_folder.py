from generate import create_npz_from_sample_folder

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_folder_dir", type=str, required=True)
    parser.add_argument("--num_fid_samples", type=int, default=50000)
    args = parser.parse_args()
    create_npz_from_sample_folder(args.sample_folder_dir, args.num_fid_samples)
    print("Done.")