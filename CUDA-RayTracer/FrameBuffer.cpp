#include "FrameBuffer.h"

#include <stdio.h>

FrameBuffer::FrameBuffer(int _width, int _height)
{
    glGenFramebuffers(1, &id);

    width = _width;
    height = _height;

    color_attachments.reserve(32);
    for (int i = 0; i < 32; i++) color_attachments.push_back(NULL);
}

void FrameBuffer::bind(int _width, int _height)
{
    glViewport(0, 0, _width, _height);
    glBindFramebuffer(GL_FRAMEBUFFER, id);
}

void FrameBuffer::unbind()
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void FrameBuffer::bind_rbo()
{
    glBindRenderbuffer(GL_RENDERBUFFER, rbo_id);
    glRenderbufferStorage(GL_RENDERBUFFER, rbo_internalformat, rbo_width, rbo_height);
    glViewport(0, 0, rbo_width, rbo_height);
}

void FrameBuffer::bind_rbo(GLsizei width, GLsizei height)
{
    glBindRenderbuffer(GL_RENDERBUFFER, rbo_id);
    glRenderbufferStorage(GL_RENDERBUFFER, rbo_internalformat, width, height);
    glViewport(0, 0, width, height);
}

void FrameBuffer::bind(aType a, int _width, int _height)
{
    unsigned int id = -1;

    if (a > Color0 && a < Color31) {
        id = color_attachments[a]->id;
    }
    else if (a == Depth) {
        id = depth_attachment->id;
    }
    else if (a == Stencil) {
        id = stencil_attachment->id;
    }
    else if (a == Main) {
        id = 0;
    }

    glViewport(0, 0, _width, _height);
    glBindFramebuffer(GL_FRAMEBUFFER, id);
}

void FrameBuffer::bind()
{
    glViewport(0, 0, width, height);
    glBindFramebuffer(GL_FRAMEBUFFER, id);
}

void FrameBuffer::bind(aType a)
{
    unsigned int id = -1;

    if (a > Color0 && a < Color31) {
        id = color_attachments[a]->id;
    }
    else if (a == Depth) {
        id = depth_attachment->id;
    }
    else if (a == Stencil) {
        id = stencil_attachment->id;
    }
    else if (a == Main) {
        id = 0;
    }

    glViewport(0, 0, width, height);
    glBindFramebuffer(GL_FRAMEBUFFER, id);
}

int FrameBuffer::attach(GLenum attachmentType, GLint internalformat, GLenum format, GLenum type, const char* name)
{
    glBindFramebuffer(GL_FRAMEBUFFER, id);

    unsigned int tex_id;
    glGenTextures(1, &tex_id);
    glBindTexture(GL_TEXTURE_2D, tex_id);

    glTexImage2D(GL_TEXTURE_2D, 0, internalformat, width, height, 0, format, type, NULL);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glGenerateMipmap(GL_TEXTURE_2D);

    glFramebufferTexture2D(GL_FRAMEBUFFER, attachmentType, GL_TEXTURE_2D, tex_id, 0);

    if (attachmentType >= GL_COLOR_ATTACHMENT0 && attachmentType <= GL_COLOR_ATTACHMENT31) {
        color_attachments[attachmentType - GL_COLOR_ATTACHMENT0] = new Attachment(name, tex_id, width, height, attachmentType, internalformat, format, type);
        glClearValue = glClearValue | GL_COLOR_BUFFER_BIT;
    }
    else if (attachmentType == GL_DEPTH_ATTACHMENT) {
        depth_attachment = new Attachment(name, tex_id, width, height, GL_DEPTH_ATTACHMENT, internalformat, format, type);
        glClearValue = glClearValue | GL_DEPTH_BUFFER_BIT;
    }
    else if (attachmentType == GL_STENCIL_ATTACHMENT) {
        stencil_attachment = new Attachment(name, tex_id, width, height, GL_STENCIL_ATTACHMENT, internalformat, format, type);
        glClearValue = glClearValue | GL_STENCIL_BUFFER_BIT;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return 1;
}

int FrameBuffer::attach_rbo(GLenum renderbuffertarget, GLenum internalformat, GLsizei width, GLsizei height)
{
    rbo_width = width;
    rbo_height = height;
    rbo_internalformat = internalformat;
    rbo_target = renderbuffertarget;

    glGenRenderbuffers(1, &rbo_id);
    glRenderbufferStorage(GL_RENDERBUFFER, rbo_internalformat, rbo_width, rbo_height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, rbo_target, GL_RENDERBUFFER, rbo_id);

    return 1;
}

void FrameBuffer::construct()
{
    glBindFramebuffer(GL_FRAMEBUFFER, id);

    vector<unsigned int> active_attachments;
    for (int i = 0; i < 32; i++) {
        if (color_attachments[i] != NULL) active_attachments.push_back(GL_COLOR_ATTACHMENT0 + i);
    }

    glDrawBuffers(active_attachments.size(), active_attachments.data());

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "Framebuffer is not complete you idiot" << std::endl;
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}
