# cp -r ../pretrained ./
docker image build -t gaugan:1.0 .
# rm -rf pretrained
docker run --rm -p 5000:5000 gaugan:1.0
