#ifndef FRAMEBUFFER_HPP
#define FRAMEBUFFER_HPP

struct FrameBuffer {
    FrameBuffer() : width(0), height(0), buffer(0) { }
    FrameBuffer(unsigned int width, unsigned int height) : width(width), height(height) {
        buffer = (Color *)malloc(sizeof(Color) * width * height);
    }
    ~FrameBuffer() {
        free(buffer);
    }

    void set(unsigned int x, unsigned int y, Color data) {
        buffer[y * width + x] = data;
    }

    Color get(unsigned int x, unsigned int y) {
        return buffer[y * width + x];
    }

    unsigned int width;
    unsigned int height;
    Color *buffer;
};

#endif //FRAMEBUFFER_HPP
