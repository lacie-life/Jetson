#include "ICM20948.h"

int main(int argc, char* argv[])
{
	IMU_EN_SENSOR_TYPE enMotionSensorType;
	IMU_ST_ANGLES_DATA stAngles;
	IMU_ST_SENSOR_DATA stGyroRawData;
	IMU_ST_SENSOR_DATA stAccelRawData;
	IMU_ST_SENSOR_DATA stMagnRawData;

	imuInit(&enMotionSensorType);
	if(IMU_EN_SENSOR_TYPE_ICM20948 == enMotionSensorType)
	{
		printf("Motion sersor is ICM-20948\n" );
	}
	else
	{
		printf("Motion sersor NULL\n");
	}

	while(1)
	{
		imuDataGet( &stAngles, &stGyroRawData, &stAccelRawData, &stMagnRawData);
		printf("\r\n /-------------------------------------------------------------/ \r\n");
		printf("\r\n Angle：Roll: %.2f     Pitch: %.2f     Yaw: %.2f \r\n",stAngles.fRoll, stAngles.fPitch, stAngles.fYaw);
		printf("\r\n Acceleration(g): X: %.3f     Y: %.3f     Z: %.3f \r\n",stAccelRawData.fX, stAccelRawData.fY, stAccelRawData.fZ);
		printf("\r\n Gyroscope(dps): X: %.3f     Y: %.3f     Z: %.3f \r\n",stGyroRawData.fX, stGyroRawData.fY, stGyroRawData.fZ);
		printf("\r\n Magnetic(uT): X: %.3f     Y: %.3f     Z: %.3f \r\n",stMagnRawData.fX, stMagnRawData.fY, stMagnRawData.fZ);
		usleep(100*1000);
	}
	return 0;
}
