#include <iostream>
#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/System/Clock.hpp>
#include <SFML/Window/Event.hpp>
#include <SFML/Graphics/CircleShape.hpp>
#include <SFML/Window.hpp>
#include <SFML/Graphics/RectangleShape.hpp>
#include <SFML/Graphics/Texture.hpp>
#include <SFML/Graphics/Sprite.hpp>

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


    uint width_texture = 400;
    uint height_texture = 400;
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