## Running the neural net with a data set 

### Create training set
```bash
g++ makeTrainingSamples.cpp -o makeTrainingSamples
./makeTrainingSamples > XORtrainingData.txt
```
### Run the neural net using the training set 
```bash
g++ neural-net-txt.cpp -o neural-net-txt
./neural-net-txt > out.txt
```
