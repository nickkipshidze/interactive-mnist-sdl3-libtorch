#include <vector>
#include <iostream>
#include <SDL3/SDL.h>
#include <SDL3_ttf/SDL_ttf.h>

#include <torch/torch.h>
#include <torch/data/datasets/mnist.h>

#include "conf.hpp"

namespace nn = torch::nn;

nn::Sequential trainMnistModel(int numEpochs=4) {
    auto dataset = torch::data::datasets::MNIST("./MNIST/raw").map(
        torch::data::transforms::Stack<>()
    );

    auto dataloader = torch::data::make_data_loader(
        std::move(dataset),
        torch::data::DataLoaderOptions()
        .batch_size(1024)
        .drop_last(true)
        .enforce_ordering(false)
    );

    auto model = nn::Sequential(
        nn::Linear(28*28, 128),
        nn::ReLU(),
        nn::Linear(128, 10)
    );

    auto optimizer = torch::optim::Adam(
        model->parameters(),
        torch::optim::AdamOptions()
        .lr(0.001)
    );

    auto criterion = nn::CrossEntropyLoss();

    for (int epoch{1}; epoch < numEpochs+1; ++epoch) {
        float lossSum = 0;
        int numBatches = 0;
        for (auto &batch : *dataloader) {
            auto X = batch.data;
            auto y = batch.target;

            optimizer.zero_grad();
            auto yLogits = model->forward(X.flatten(1));
            auto loss = criterion->forward(yLogits, y);
            loss.backward();
            optimizer.step();
            
            lossSum += loss.item<float>();
            numBatches++;

            std::cout << "* Epoch [" << epoch << "/" << numEpochs << "] Batch [" << numBatches << "] AVG Loss: " << lossSum / numBatches << "\r";
            std::flush(std::cout);
        }
        std::cout << std::endl;
    }

    return model;
}

void renderGrid(SDL_Renderer *renderer, SDL_FPoint size, SDL_FPoint offset, std::vector<float> &data, int rows, int columns, float padding=3.0f) {
    SDL_FRect bgrect = {offset.x-padding, offset.y-padding, columns*(size.x+1)+padding*2, rows*(size.y+1)+padding*2};
    SDL_SetRenderDrawColor(renderer, 66, 66, 66, 255);
    SDL_RenderFillRect(renderer, &bgrect);

    for (int y{0}; y < rows; ++y) {
        for (int x{0}; x < columns; ++x) {
            int index = (y*columns)+x;
            Uint8 value = 255.0f*data[index];
            SDL_FRect cell = {x*(size.x+1)+offset.x, y*(size.y+1)+offset.y, size.x, size.y};
            SDL_SetRenderDrawColor(renderer, value, value, value, 255);
            SDL_RenderFillRect(renderer, &cell);
        }
    }
}

void renderText(SDL_Renderer *renderer, TTF_Font *font, const char *text, SDL_FPoint position, SDL_Color color) {
    SDL_Surface *surface = TTF_RenderText_Solid(font, text, 0, color);
    SDL_Texture *texture = SDL_CreateTextureFromSurface(renderer, surface);
    SDL_FRect dstrect = {position.x, position.y, (float)texture->w, (float)texture->h};
    SDL_RenderTexture(renderer, texture, nullptr, &dstrect);
    SDL_DestroyTexture(texture);
    SDL_DestroySurface(surface);
}

void renderPredictions(SDL_Renderer *renderer, torch::Tensor &predictions, SDL_FPoint offset, TTF_Font *font, float maxWidth=347.0f, float height=17.65f, float gaps=18.0f) {
    for (int i{0}; i < predictions.size(0); ++i) {
        SDL_FRect barrect = {
            offset.x, (height+gaps)*i+offset.y,
            maxWidth, height
        };

        SDL_SetRenderDrawColor(renderer, 31, 31, 31, 255);
        SDL_RenderFillRect(renderer, &barrect);
        
        barrect.w = maxWidth*predictions[i].item<float>();
        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
        SDL_RenderFillRect(renderer, &barrect);

        // Smart trick since it's 0-9
        char number[] = {char(i+'0'), '\0'};
        renderText(renderer, font, number, {barrect.x-31, barrect.y-9}, {139, 139, 139});
    }
}

void initVector(std::vector<float> &data, float value, int size) {
    while (data.size() < size)
        data.push_back(value);
    for (int i{0}; i < size; ++i)
        data[i] = value;
}

SDL_Point drawOnGrid(std::vector<float> &data, SDL_FPoint size, SDL_FPoint offset, int rows, int columns) {
    SDL_FPoint mousePos;
    SDL_MouseButtonFlags flags = SDL_GetMouseState(&mousePos.x, &mousePos.y);
    if (!flags & SDL_BUTTON_MASK(SDL_BUTTON_LEFT)) return {-1, -1};
    
    int x = (int)((mousePos.x - offset.x) / size.x);
    int y = (int)((mousePos.y - offset.y) / size.y);

    if (x < 0 || x >= columns || y < 0 || y >= rows)
        return {-1, -1};

    return {x, y};
}

void safeAdd(std::vector<float> &data, int columns, SDL_Point position, float value) {
    if (position.x < 0 || position.y < 0 || position.x > columns-1 || position.y > data.size()/columns-1) return;
    int index = (position.y*columns)+position.x;
    if (index > 0 && index < data.size())
        data[index] = SDL_clamp(data[index]+value, 0, 1);
}

// Started dev 2:26 PM
// Paused dev 4:14 PM
// Continued dev 4:32 PM
// Paused dev 6:21 PM
// Continued dev 8:25 PM
// Paused dev 8:55 PM
int main(int argc, char *argv[]) {
    SDL_Init(SDL_INIT_VIDEO);
    TTF_Init();

    auto model = trainMnistModel(6);
    model->eval();

    SDL_Window *window = SDL_CreateWindow("Interactive MNIST classifier", conf::winsize.x, conf::winsize.y, 0);
    SDL_Renderer *renderer = SDL_CreateRenderer(window, nullptr);

    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
    SDL_SetRenderLogicalPresentation(renderer, conf::resolution.x, conf::resolution.y, SDL_LOGICAL_PRESENTATION_LETTERBOX);

    TTF_Font *fontInterBold = TTF_OpenFont("./Inter-ExtraBold.ttf", 24);

    constexpr int rows = 28, columns = 28;
    std::vector<float> data;
    data.reserve(rows*columns);
    initVector(data, 0.0f, rows*columns);

    SDL_FPoint gridOffsets = {75+3, 71+3};
    torch::Tensor predictions = torch::zeros(10);

    Uint64 clockNow = SDL_GetPerformanceCounter(), clockLast = clockNow;

    bool running = true;
    while (running) {
        SDL_Delay((1.0f/conf::FPS)*1000);
        clockNow = SDL_GetPerformanceCounter();
        float deltaTime = (float)(clockNow - clockLast) / SDL_GetPerformanceFrequency();
        clockLast = clockNow;

        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
                case SDL_EVENT_QUIT:
                    running = false;
                    break;
                case SDL_EVENT_KEY_DOWN:
                    if (event.key.scancode == SDL_SCANCODE_Q)
                        running = false;
                    if (event.key.scancode == SDL_SCANCODE_C) {
                        predictions = torch::zeros(10);
                        initVector(data, 0.0f, rows*columns);
                    }
                    break;
            }
        }

        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        renderText(renderer, fontInterBold, "Draw your digit", {75, 33}, {255, 255, 255});
        renderText(renderer, fontInterBold, "Predictions (Softmax)", {582, 33}, {255, 255, 255});
        renderText(renderer, fontInterBold, "Press C to clear", {75, 449}, {59, 59, 59});
        renderText(renderer, fontInterBold, "Q to quit", {343, 449}, {59, 59, 59});

        bool gridChanged = false;
        SDL_Point p = drawOnGrid(data, {13, 13}, gridOffsets, rows, columns);
        if (p.x != -1 && p.y != -1) {
            safeAdd(data, columns, {p.x, p.y}, 32.0f * deltaTime);
            safeAdd(data, columns, {p.x+1, p.y}, 32.0f * deltaTime);
            safeAdd(data, columns, {p.x, p.y+1}, 32.0f * deltaTime);
            gridChanged = true;
        }

        if (gridChanged) {
            torch::InferenceMode guard;
            auto yLogits = model->forward(torch::tensor(data).unsqueeze(0));
            auto yPred = torch::softmax(yLogits.squeeze(), 0);
            predictions = yPred;
        }
        renderPredictions(renderer, predictions, {613, 89}, fontInterBold);

        SDL_FRect sep{512, 0, 1, 512};
        SDL_SetRenderDrawColor(renderer, 66, 66, 66, 255);
        SDL_RenderFillRect(renderer, &sep);

        renderGrid(renderer, {12, 12}, gridOffsets, data, rows, columns);

        // Shameless plug, because my projects *have* been stolen before (true story)
        renderText(renderer, fontInterBold, "Made by Nick Kipshidze", {680, 449}, {59, 59, 59});

        SDL_RenderPresent(renderer);
    }

    TTF_CloseFont(fontInterBold);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    TTF_Quit();
    SDL_Quit();

    return 0;
}
