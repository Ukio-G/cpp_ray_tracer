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
    Ray computeForPixel(unsigned int x, unsigned int y);
    std::pair<AGeomerty *, double> findNearestIntersect(const Ray &ray);


private:
    ApplicationData data;

    Color computeColorFromLight(std::pair<AGeomerty *, double> intersectedShape);
};


#endif
