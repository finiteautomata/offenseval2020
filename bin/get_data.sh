# Download the zips for these languages
wget -nc -O data/arabic.zip https://www.dropbox.com/sh/qfootyofgiywjjl/AADMWDIaER71EUMcg3Pd0n2Qa/Arabic.zip?dl=0
wget -nc -O data/danish.zip https://www.dropbox.com/sh/qfootyofgiywjjl/AAC1AGvb_6OOg8KeyWI_QIlZa/Danish.zip?dl=0
wget -nc -O data/greek.zip https://www.dropbox.com/sh/qfootyofgiywjjl/AACDTE3rHIbzN7ZBRurPQcJna/Greek.zip?dl=0
wget -nc -O data/turkish.zip https://www.dropbox.com/sh/qfootyofgiywjjl/AAC65KmUVmuE1gM1DHgoWtI5a/Turkish.zip?dl=0

# English data
gdown -O data/english-task-a.zip https://drive.google.com/uc?id=10j4FeNTTh9yeOwtpo2mdB7YQS9urNnLi
gdown -O data/english-task-b.zip https://drive.google.com/uc?id=1grFkLmYts5Dw_yJVnUS-353Uc14FBf5D
gdown -O data/english-task-c.zip https://drive.google.com/uc?id=1fs1PUvvfBqFdf9QOXCtF2u9gMrHZXkq0
gdown -O data/README_data_en.txt https://drive.google.com/uc?id=1Sxlvu2fLX9HWY28bqcW6NAcP6Fb3TNNd

# Unzip them
cd data
FILES=*.zip
PASSWORD=sem2020-t12

for file in $FILES
do
  unzip -P $PASSWORD $file
done

mkdir Arabic
mv offenseval-ar-* Arabic/
mv readme-data-ar.txt Arabic/
mv README_data_en.txt English/
mv task_* English
rm *.zip
