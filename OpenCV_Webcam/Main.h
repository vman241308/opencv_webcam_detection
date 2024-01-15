#pragma once
#include "opencv2\opencv.hpp"
#include <iostream>
#include <ctime>

int main(int argc, char* argv[]);

// Handle commandline arguments
void handleArgs(int argc, char* argv[]);

// Timestamp an image
void timestamp(cv::Mat * image_to_stamp);

// Print a message on a image
void drawMessage(cv::Mat inframe, cv::OutputArray outframe, char message[]);

// Draw window update logic
int updateWindow(cv::VideoCapture * capture_stream);