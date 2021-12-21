#ifndef APPLICATIONLOGIC_HPP
#define APPLICATIONLOGIC_HPP

#include "Parser.hpp"
#include "ApplicationData.hpp"

class ApplicationLogic {
public:
    ApplicationLogic();

    void initFromFile(const std::string & filename);
    void renderFrameBuffer();
    const FrameBuffer& getFrameBuffer();
    void computeRayForPixel(unsigned int x, unsigned int y);
    std::pair<AGeomerty *, double> findNearestIntersect(const Ray &ray);
    void swapFrameBufferToSfImage(sf::Image &image);

private:
    ApplicationData data;

    Color computeColorFromLight(std::pair<AGeomerty *, double> intersectedShape, Ray &ray);

    bool lightAvailable(std::pair<AGeomerty *, double> pair1, Vec3d vector, LightSource &source);
};


#endif
