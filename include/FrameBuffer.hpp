#ifndef FRAMEBUFFER_HPP
#define FRAMEBUFFER_HPP

struct FrameBuffer {
    FrameBuffer() : width(0), height(0), buffer(nullptr) { }
    FrameBuffer(unsigned int width, unsigned int height) : width(width), height(height) {
        size_t buffer_size = sizeof(Color) * width * height;
        buffer = (Color *)malloc(buffer_size);
    }
    ~FrameBuffer() {
    }

public:
    void set(unsigned int x, unsigned int y, Color data) {
        buffer[y * width + x] = data;
    }

    Color get(unsigned int x, unsigned int y) {
        return buffer[y * width + x];
    }

    unsigned int width;
    unsigned int height;
// private:
    Color *buffer;
};

#endif //FRAMEBUFFER_HPP
