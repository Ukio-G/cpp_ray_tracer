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


int main()
{
    sf::ContextSettings settings;

    settings.depthBits = 24;
    settings.stencilBits = 8;
    settings.antialiasingLevel = 4;
    settings.majorVersion = 3;
    settings.minorVersion = 0;

    sf::RenderWindow window(sf::VideoMode(1500, 1000), "OpenGL", sf::Style::Default, settings);
    window.setVerticalSyncEnabled(false);

    window.setActive(true);

    Parser p;


    std::fstream file("../config_rt.txt");
    char buf[512];
    while (file.getline(buf,512)) {
        //std::cout << buf << std::endl;

        auto item = p.parse_line(buf);

        if (item.has_value()) {


            if (item->index() == 0) {           // AGeometry


                std::shared_ptr<AGeomerty> geomerty = std::get<0>(*item);
                if (Sphere * sphere = dynamic_cast<Sphere*>(geomerty.get()))
                    std::cout << "Sphere: " << *sphere << std::endl;
                else if (Plane * plane = dynamic_cast<Plane*>(geomerty.get()))
                    std::cout << "Plane: " << *plane << std::endl;
                else if (Square * square = dynamic_cast<Square*>(geomerty.get()))
                    std::cout << "Square: " << *square << std::endl;
                else if (Triangle * triangle = dynamic_cast<Triangle*>(geomerty.get()))
                    std::cout << "Triangle: " << *triangle << std::endl;
                else if (Cylinder * cylinder = dynamic_cast<Cylinder*>(geomerty.get()))
                    std::cout << "Cylinder: " << *cylinder << std::endl;


            } else if (item->index() == 1) {    // Camera
                std::cout << "Camera: " << *(std::get<1>(*item).get()) << std::endl;
            } else if (item->index() == 2) {    // ALight

                std::shared_ptr<ALight> light = std::get<2>(*item);
                if (AmbientLight * ambientLight = dynamic_cast<AmbientLight*>(light.get()))
                    std::cout << "AmbientLight: " << *ambientLight << std::endl;
                else if (LightSource * lightSource = dynamic_cast<LightSource*>(light.get()))
                    std::cout << "LightSource: " << *lightSource << std::endl;

            } else if (item->index() == 3) {    // Vec2i
                auto resolution = *(std::get<3>(*item).get());
                std::cout << "Window resolution: " << resolution[0] << " " << resolution[1] << std::endl;
            }



        }
        else {
            std::cout << "Can't recognize line" << std::endl;
        }

        memset(buf, 0, 512);
    }

    exit(0);

    unsigned int width_texture = 400;
    unsigned int height_texture = 400;
    /* Create texture */
    sf::Texture texture_;
    texture_.create(width_texture,height_texture);

    /* Create image */
    sf::Image image;
    image.create(width_texture, height_texture, sf::Color::White);

    /* Set image data */
    for (int i = 0; i < width_texture; ++i) {
        for (int j = 0; j < height_texture; ++j) {
            image.setPixel(i, j, sf::Color(i%255, j%255, (i+j)%255));
        }
    }

    /* Update texture */
    texture_.update(image);

    /* Create sprite and init from texture */
    sf::Sprite sprite;
    sprite.setTexture(texture_);

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