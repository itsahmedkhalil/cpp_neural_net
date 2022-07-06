## Running the neural net with a data set 

### Create training set
```bash
g++ makeTrainingSamples.cpp -o makeTrainingSamples
./makeTrainingSamples > XORtrainingData.txt
```
### Run the neural net using the training set 
```bash
g++ neural-net.cpp -o neural-net
./neural-net > out.txt
```

### Test
https://youtu.be/ndAfWKmKF34