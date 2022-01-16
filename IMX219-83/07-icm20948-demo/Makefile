ICM20948_Demo:main.o ICM20948.o
	gcc -Wall -o ICM20948_Demo main.o ICM20948.o -lm -std=gnu99
main.o: main.c ICM20948.h
	gcc -Wall -c main.c -lm -std=gnu99
ICM20948.o: ICM20948.c ICM20948.h
	gcc -Wall -c ICM20948.c -lm -std=gnu99	
clean:
	rm main.o ICM20948.o ICM20948_Demo