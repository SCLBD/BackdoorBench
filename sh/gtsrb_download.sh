# please download the following files and put them in ../data folder
wget -P ../data https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip --no-check-certificate
wget -P ../data https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip --no-check-certificate
wget -P ../data https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip --no-check-certificate
mkdir ../data/gtsrb;
mkdir ../data/gtsrb/Train;
mkdir ../data/gtsrb/Test;
mkdir ../data/temps;
unzip ../data/GTSRB_Final_Training_Images.zip -d ../data/temps/Train;
unzip ../data/GTSRB_Final_Test_Images.zip -d ../data/temps/Test;
mv ../data/temps/Train/GTSRB/Final_Training/Images/* ../data/gtsrb/Train;
mv ../data/temps/Test/GTSRB/Final_Test/Images/* ../data/gtsrb/Test;
unzip ../data/GTSRB_Final_Test_GT.zip -d ../data/gtsrb/Test/;
rm -r ../data/temps;
rm ../data/*.zip;
echo "Download Completed";
