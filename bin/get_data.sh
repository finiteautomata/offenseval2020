# Download the zips for these languages
wget -nc -O data/arabic.zip https://www.dropbox.com/sh/qfootyofgiywjjl/AADMWDIaER71EUMcg3Pd0n2Qa/Arabic.zip?dl=0
wget -nc -O data/danish.zip https://www.dropbox.com/sh/qfootyofgiywjjl/AAC1AGvb_6OOg8KeyWI_QIlZa/Danish.zip?dl=0
wget -nc -O data/greek.zip https://www.dropbox.com/sh/qfootyofgiywjjl/AACDTE3rHIbzN7ZBRurPQcJna/Greek.zip?dl=0
wget -nc -O data/turkish.zip https://www.dropbox.com/sh/qfootyofgiywjjl/AAC65KmUVmuE1gM1DHgoWtI5a/Turkish.zip?dl=0
mkdir data/olid
wget -nc -O data/olid/olid.zip https://sites.google.com/site/offensevalsharedtask/olid/OLIDv1.0.zip?attredirects=0&d=1
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
mkdir English
mv offenseval-ar-* Arabic/
mv readme-data-ar.txt Arabic/
mv README_data_en.txt English/
mv task_* English
rm *.zip
# Get OLID and unzip it
cd olid
unzip olid.zip
rm olid.zip
cd ../..

######################################
###          TEST DATA             ###
######################################
# Run from here to only retrieve test data
#
cd data/

download_and_unzip(){
    wget -nc -O $1/test$3.zip $2
    cd $1 && unzip -o test$3.zip && rm test$3.zip && cd ..
}

download_and_unzip Turkish https://competitions.codalab.org/my/datasets/download/10cdc735-ee5e-4ab9-8663-e99b4aae9077
download_and_unzip Greek https://competitions.codalab.org/my/datasets/download/b6bef21a-87fd-4ea2-8a89-ede45f355c7b
download_and_unzip Danish https://competitions.codalab.org/my/datasets/download/6bdaf54f-e52b-4bd0-a11c-c8a3ed9dc77d
download_and_unzip English https://competitions.codalab.org/my/datasets/download/75007c72-09e9-4f4e-ac95-d2c5ff5eda62
download_and_unzip English https://competitions.codalab.org/my/datasets/download/4264b414-30d3-452f-ad5c-ab4679aa63fd _b
wget -nc -O Arabic/test.tsv https://competitions.codalab.org/my/datasets/download/9d850022-55e4-4311-be26-50de0094d16a

mv Greek/testset_taska.tsv Greek/test.tsv
mv Arabic/test.tsv Arabic/test.tsv
mv Turkish/offenseval-tr-testset-v1.tsv Turkish/test.tsv
mv Danish/offenseval-da-test-v1-nolabels.tsv Danish/test.tsv
mv English/test_a_tweets.tsv English/test.tsv
mv English/test_b_tweets.tsv English/test_b.tsv
