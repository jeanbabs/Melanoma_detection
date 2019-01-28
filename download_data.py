# IMPORTANT
# You have to run pip install google_drive_downloader 


import os
from google_drive_downloader import GoogleDriveDownloader as gdd
def main(dest_dir = 'data'):
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    output_file = os.path.join(dest_dir, "data.zip")
    
    print("Starting download...")
    gdd.download_file_from_google_drive(file_id='1z1uVQpQvtjJocgbANCxfFaEVGEzYhnOc', dest_path=output_file, unzip=True)
    print("=> File saved as {}".format(output_file))
    



if __name__ == '__main__':
    main()