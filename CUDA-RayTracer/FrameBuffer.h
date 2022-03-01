#ifndef FBO_H
#define FBO_H

#include "GLCommon.h"

#include "Attachment.h"
#include "Texture.h"

#include <vector>

enum aType
{
	Color0,
	Color1,
	Color2,
	Color3,
	Color4,
	Color5,
	Color6,
	Color7,
	Color8,
	Color9,
	Color10,
	Color11,
	Color12,
	Color13,
	Color14,
	Color15,
	Color16,
	Color17,
	Color18,
	Color19,
	Color20,
	Color21,
	Color22,
	Color23,
	Color24,
	Color25,
	Color26,
	Color27,
	Color28,
	Color29,
	Color30,
	Color31,
	Depth,
	Stencil,
	Main
};

class FrameBuffer {

public:
	unsigned int id;
	unsigned int rbo_id;
	Attachment* depth_attachment;
	Attachment* stencil_attachment;
	std::vector<Attachment*> color_attachments;

	int width;
	int height;

	int rbo_width;
	int rbo_height;
	int rbo_internalformat;
	int rbo_target;

	int glClearValue = 1;

	FrameBuffer(int _width, int _height);
	void bind(aType a);
	void bind(aType a, int _width, int _height);
	void bind();
	void bind(int _width, int _height);
	void unbind();

	void bind_rbo();
	void bind_rbo(GLsizei width, GLsizei height);
	//int attach(GLenum attachment, GLint internalformat, GLenum format, GLenum type);
	int attach(GLenum attachmentType, GLint internalformat, GLenum format, GLenum type, const char* name = "no_name");
	int attach_rbo(GLenum renderbuffertarget, GLenum internalformat, GLsizei width, GLsizei height);
	void construct();

	unsigned int getID() { return id; }

private:

};
#endif