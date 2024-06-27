pip install -r requirements.txt

mkdir data
cd data
wget -O sentences.tar.gz https://datadryad.org/api/v2/files/2547369/download
tar -xvzf sentences.tar.gz
cd ..

python scripts/formatCompetitionData.py
