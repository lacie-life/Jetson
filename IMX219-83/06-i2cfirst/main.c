#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdint.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <linux/i2c.h>
#include <linux/i2c-dev.h>

#define IIC_Dev  "/dev/i2c-1"
int fd;

uint8_t I2C_ReadOneByte(uint8_t DevAddr, uint8_t RegAddr)
{
  uint8_t u8Ret;
  if (ioctl(fd, I2C_SLAVE, DevAddr) < 0)
  {
    printf("Failed to acquire bus access and/or talk to slave.\n");
    return 0;
  }
  write(fd, &RegAddr,1);
  read(fd, &u8Ret, 1);
  return u8Ret;
}

int main(int argc, char* argv[])
{
    if ((fd = open(IIC_Dev, O_RDWR)) < 0)
    {
        printf("Failed to open the i2c bus.\n");
        return -1;
    }

    uint8_t u8DevAddr,u8RegAddr, u8RegVel;
    u8DevAddr = 0x68;
    u8RegAddr = 0x00;
    u8RegVel = I2C_ReadOneByte(u8DevAddr, u8RegAddr);
    printf("i2c-dev[0x%02x] : register[0x%02x] value is [0x%02x]\n", u8DevAddr,u8RegAddr,u8RegVel);
    
    close(fd);
    return 0;
}
