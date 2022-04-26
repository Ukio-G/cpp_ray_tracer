#ifndef APPLICATIONLOGIC_HPP
#define APPLICATIONLOGIC_HPP

#include "Parser.hpp"
#include "ApplicationData.hpp"

enum RenderMode {
    CUDA,
    CPU
};

class ApplicationLogic {
public:
    ApplicationLogic();

    void initFromFile(const std::string& filename);
    void initFromFileContinuous(const std::string & filename);
    void renderFrameBuffer(RenderMode renderMode);

    const FrameBuffer& getFrameBuffer();
    Ray computeRayForPixel(unsigned int x, unsigned int y);
    std::pair<AGeomerty *, double> findNearestIntersect(const Ray &ray);
    void swapFrameBufferToSfImage(sf::Image &image);

    ApplicationData data;
    cuAppData* cuData;
private:
    void renderFrameBufferCPU();
    void renderFrameBufferCUDA();

    Color computeColorFromLight(std::pair<AGeomerty *, double> intersectedShape, Ray &ray);

    bool lightAvailable(std::pair<AGeomerty *, double> pair1, Vec3d vector, LightSource &source);
};


#endif
