#pragma once 

#include "GLCommon.h"

struct Attachment {

    unsigned int id;
    const char* name;
    int width;
    int height;

    GLenum attachmentType;
    GLint internalformat;
    GLenum format;
    GLenum type;

    Attachment(const char* _name, unsigned int _id, int _width, int _height, GLenum _attachmentType, GLint _internalformat, GLenum _format, GLenum _type) {
        id = _id;
        name = _name;
        width = _width;
        height = _height;
        attachmentType = _attachmentType;
        internalformat = _internalformat;
        format = _format;
        type = _type;
    }

    void bind() {
        glBindTexture(GL_TEXTURE_2D, id);
        glTexImage2D(GL_TEXTURE_2D, 0, internalformat, width, height, 0, format, type, NULL);
    }
};
