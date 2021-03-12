#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <stdio.h>
#include <stdarg.h>       

int main(int argc, char * argv[])
{
    rs2::log_to_console(RS2_LOG_SEVERITY_ERROR);

    rs2::colorizer color_map;
    rs2::rates_printer printer;
    rs2::pipeline pipe;

    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    pipe.start(cfg);

    std::map<int, rs2::frame> render_frames;
    std::vector<rs2::frame> new_frames;

    cv::Mat image;

    while (1) 
    {
        rs2::frameset fs = pipe.wait_for_frames();;

        rs2::frame color_frame = fs.get_color_frame();
        cv::Mat color(cv::Size(640, 480), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
    //    cv::cvtColor(color, image, cv::COLOR_BGR2RGB);
        cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
        cv::imshow("Display Image", color);

        char c = (char)cv::waitKey(25);
        if(c == 27)
            exit(0);
    }

    return EXIT_SUCCESS;
}
