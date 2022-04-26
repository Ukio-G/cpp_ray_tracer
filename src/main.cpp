#include <iostream>
#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/System/Clock.hpp>
#include <SFML/Window/Event.hpp>
#include <SFML/Graphics/CircleShape.hpp>
#include <SFML/Window.hpp>
#include <SFML/Graphics/RectangleShape.hpp>
#include <SFML/Graphics/Texture.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include <optional>
#include <Parser.hpp>
#include <fstream>
#include <ApplicationLogic.hpp>
#include <chrono>
#include "../CudaAppData.cuh"
#include "geometry/Sphere.hpp"
#include "../CheckerStructSize.hpp"
#include "geometry/AGeomerty.hpp"
#include "../TimeLogger.h"


int main(int argc, char ** argv)
{
    TimeLogger::getInstance()->StartMeasure("Global");
    checkStructSize();

    sf::ContextSettings settings;

    ApplicationLogic application;
    std::cout << argc << std::endl;
    std::string filename = (argc == 1) ? "../config_rt.txt" : argv[1];
    TimeLogger::getInstance()->StartMeasure("CPUInitData");
    application.initFromFile(filename);
    TimeLogger::getInstance()->StopMeasure("CPUInitData");

    application.initFromFileContinuous(filename);

    /* Start Configure window context */
    settings.depthBits = 24;
    settings.stencilBits = 8;
    settings.antialiasingLevel = 4;
    settings.majorVersion = 3;
    settings.minorVersion = 0;
    sf::RenderWindow window(sf::VideoMode(application.getFrameBuffer().width, application.getFrameBuffer().height), "Raytrace SFML", sf::Style::Default, settings);
    window.setVerticalSyncEnabled(false);
    window.setActive(true);
    window.setFramerateLimit(30);
    /* End Configure window context */

    unsigned int width_texture = application.getFrameBuffer().width;
    unsigned int height_texture = application.getFrameBuffer().height;

#ifdef CUDA_RENDER
    /* CUDA start init */
    TimeLogger::getInstance()->StartMeasure("CudaInitData");
    auto initStruct = application.data.generateCudaInitStruct();
    TimeLogger::getInstance()->StopMeasure("CudaInitData");

    TimeLogger::getInstance()->StartMeasure("CudaInitCopy");
    cuAppData cudaAppData(&initStruct);
    TimeLogger::getInstance()->StopMeasure("CudaInitCopy");

    /* CUDA end init */

    /* Render scene. Magic start from this point */
    application.cuData = &cudaAppData;
#endif
    application.renderFrameBuffer(CPU);

    /* Create texture */
    sf::Texture texture_;
    texture_.create(width_texture,height_texture);

    /* Create image */
    sf::Image image;

    /* Load rendered data from our frame buffer to image */
    application.swapFrameBufferToSfImage(image);

    /* Update texture */
    texture_.update(image);

    /* Create sprite and init from texture */
    sf::Sprite sprite;
    sprite.setTexture(texture_);
    TimeLogger::getInstance()->StopMeasure("Global");

    TimeLogger::PrintAllMeasures();

    // run the main loop
    bool running = true;
    while (running)
    {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                running = false;
        }
        window.clear();

        // Render shapes
        window.draw(sprite);
        window.display();
    }
    std::cout << "\n";
	return 0;
}