#include "HelloWorldScene.h"
#include "AppMacros.h"

#include "nanovg.h"

#if (CC_TARGET_PLATFORM == CC_PLATFORM_ANDROID) || (CC_TARGET_PLATFORM == CC_PLATFORM_IOS)
#define NANOVG_GLES2_IMPLEMENTATION
#else
#define NANOVG_GL2_IMPLEMENTATION
#endif

#include "nanovg_gl.h"
#include "nanovg_gl_utils.h"

#include "cprocessing.hpp"

USING_NS_CC;

using namespace cprocessing;

namespace cprocessing {
    int mouseX;
    int mouseY;
    int width;
    int height;
    float frameCount;
}

NVGcontext* vg;

std::vector<float> angle, rad, speed, x, y, diam;
std::vector<int> nbConnex;
int nbPts;
static int s_RADIUS = 30;


void initialize(){
    nbPts = random(30,150);
    angle = std::vector<float>(nbPts);
    rad = std::vector<float>(nbPts);
    speed = std::vector<float>(nbPts);
    x = std::vector<float>(nbPts);
    y = std::vector<float>(nbPts);
    diam = std::vector<float>(nbPts);
    nbConnex = std::vector<int>(nbPts);
    for (int i = 0; i<nbPts; i++) {
        angle[i] = random(0.0,TWO_PI);
        rad[i] = random(1, 5) * s_RADIUS;
        speed[i] = random(-.01, .01);
        x[i] = width/2;
        y[i] = height/2;
        nbConnex[i] = 0;
        diam[i] = 0;
    }
}

void setup() {
    initialize();
}


void mousePressed() {
    initialize();
}

float ease(float variable, float target, float easingVal) {
    float d = target - variable;
    if (abs(d)>1) variable+= d*easingVal;
    return variable;
}


Scene* HelloWorld::scene()
{
    
    // 'scene' is an autorelease object
    auto scene = Scene::create();
    
    // 'layer' is an autorelease object
    HelloWorld *layer = HelloWorld::create();

    // add layer as a child to scene
    scene->addChild(layer);

    // return the scene
    return scene;
}


// on "init" you need to initialize your instance
bool HelloWorld::init()
{
    //////////////////////////////
    // 1. super init first
    if ( !Layer::init() )
    {
        return false;
    }
    
    auto visibleSize = Director::getInstance()->getVisibleSize();
    auto origin = Director::getInstance()->getVisibleOrigin();

    auto closeItem = MenuItemImage::create(
                                           "CloseNormal.png",
                                           "CloseSelected.png",
                                           CC_CALLBACK_1(HelloWorld::menuCloseCallback,this));
    
    closeItem->setPosition(origin + Vec2(visibleSize) - Vec2(closeItem->getContentSize() / 2));
    
    // create menu, it's an autorelease object
    auto menu = Menu::create(closeItem, NULL);
    menu->setPosition(Vec2::ZERO);
    this->addChild(menu, 1);
    
    
    width = visibleSize.width;
    height = visibleSize.height;

    
    auto listener = EventListenerTouchAllAtOnce::create();
    listener->onTouchesEnded = CC_CALLBACK_2(HelloWorld::onTouchesEnded, this);
    _eventDispatcher->addEventListenerWithSceneGraphPriority(listener, this);
    

    Director::getInstance()->setDisplayStats(true);
    
#if (CC_TARGET_PLATFORM == CC_PLATFORM_ANDROID) || (CC_TARGET_PLATFORM == CC_PLATFORM_IOS)
    vg = nvgCreateGLES2(NVG_ANTIALIAS | NVG_STENCIL_STROKES | NVG_DEBUG);
#else
    vg = nvgCreateGL2(NVG_ANTIALIAS | NVG_STENCIL_STROKES | NVG_DEBUG);
#endif
    initialize();
    
    this->scheduleUpdate();
    t = 0;
    
    return true;
}

void HelloWorld::onTouchesEnded(const std::vector<Touch*>& touches, Event* event)
{
    initialize();
}

CustomCommand _customCommand;

void HelloWorld::draw(Renderer *renderer, const Mat4 &transform, uint32_t flags)
{
    _customCommand.init(_globalZOrder);
    _customCommand.func = CC_CALLBACK_0(HelloWorld::onDraw, this, transform, flags);
    renderer->addCommand(&_customCommand);
}



void HelloWorld::onDraw(const Mat4 &transform, uint32_t flags)
{
    auto origin =  Director::getInstance()->getVisibleOrigin();
    
    nvgBeginFrame(vg, width, height, 1);
    
    nvgBeginPath(vg);
    
    // draw background
     nvgRect(vg, 0,0, width,height);
     nvgFillColor(vg, nvgRGBA(255,255,255,255));
     nvgFill(vg);
     
     nvgStrokeColor(vg, nvgRGBA(0,0,0,50));
     for (int i=0; i<nbPts-1; i++) {
         for (int j=i+1; j<nbPts; j++) {
             if (dist(x[i], y[i], x[j], y[j])<RADIUS+10) {
                 nvgBeginPath(vg);
                 nvgMoveTo(vg, x[i], y[i]);
                 nvgLineTo(vg, x[j], y[j]);
                 nvgStroke(vg);
                 nbConnex[i]++;
                 nbConnex[j]++;
             }
         }
     }
 
     for (int i=0; i<nbPts; i++) {
         angle[i] += speed[i];
         x[i] = ease(x[i], width/2 + cos(angle[i]) * rad[i], 0.1);
         y[i] = ease(y[i], height/2 + sin(angle[i]) * rad[i], 0.1);
         diam[i] = ease(diam[i], min(nbConnex[i], 7)*max(0.5,(rad[i]/RADIUS/5.0)), 0.1);
         
         nvgBeginPath(vg);
         nvgFillColor(vg, nvgRGBA(0,0,0,100));
         nvgEllipse(vg, x[i], y[i], diam[i] + 3, diam[i] + 3);
         nvgFill(vg);
         
         nvgBeginPath(vg);
         nvgFillColor(vg, nvgRGBA(0,0,0,255));
         nvgEllipse(vg, x[i], y[i], diam[i], diam[i]);
         nvgFill(vg);
         
         nbConnex[i] = 0;
     }
    
    nvgBeginPath(vg);
    nvgMoveTo(vg, 0,0);
    nvgLineTo(vg, 100, 40);
    nvgStrokeColor(vg, nvgRGB(255,255,255));
    nvgStrokeWidth(vg, 3);
    nvgStroke(vg);
    
    nvgEndFrame(vg);
    
    GL::bindTexture2D(0);
    GL::enableVertexAttribs(GL::VERTEX_ATTRIB_FLAG_NONE);

    GL::useProgram(0);
}

void HelloWorld::update(float dt)
{
    t += dt;
    frameCount = 1/dt;
}

void HelloWorld::menuCloseCallback(Ref* sender)
{
    
}


