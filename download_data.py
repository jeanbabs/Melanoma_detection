# IMPORTANT
# You have to run pip install google_drive_downloader 


import os
from google_drive_downloader import GoogleDriveDownloader as gdd
def main(dest_dir = 'data'):
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    output_file = os.path.join(dest_dir, "data_test.zip")
    
    print("Starting downloading test data set...")
    gdd.download_file_from_google_drive(file_id='1_Os49PsPTj3dX1i5XC9A8h6whbdNVG0S', dest_path=output_file, unzip=True)
    print("=> File saved as {}".format(output_file))

    output_file2 = os.path.join(dest_dir, "data_train.zip")

    print("Starting downloading train data set...")
    gdd.download_file_from_google_drive(file_id='1iMF9cMwpp0zFkEgqyPq2DCBA_hV04cz_', dest_path=output_file2, unzip=True)
    print("=> File saved as {}".format(output_file2))
   
    


if __name__ == '__main__':
    main()
