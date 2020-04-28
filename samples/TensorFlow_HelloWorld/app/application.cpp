/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <SmingCore.h>
#include <HardwarePWM.h>

#include <TensorFlowLite.h>

#include "constants.h"
#include "output_handler.h"
#include "sine_model_data.h"
#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// TensorFlow specific code
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

// Create an area of memory to use for input, output, and intermediate arrays.
// Finding the minimum value for your model may require some trial and error.
constexpr int kTensorArenaSize = 2 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
// -- End of TensorFlow specific code

Timer procTimer;

const int ledPin = 2; // GPIO2

// Track whether the function has run at least once
bool initialized = false;

//uint8_t pins[8] = {ledPin}; // List of pins that you want to connect to pwm
//HardwarePWM pwm(pins, 1);

// Animates a dot across the screen to represent the current x and y values
void handleOutput(tflite::ErrorReporter* error_reporter, float x_value, float y_value)
{
	// Do this only once
	if(!initialized) {
		// Set the LED pin to output
		TF_LITE_REPORT_ERROR(error_reporter, "Set the LED pin to output");
		//    pinMode(ledPin, OUTPUT);
		initialized = true;
	}

	// Calculate the brightness of the LED such that y=-1 is fully off
	// and y=1 is fully on. The LED's brightness can range from 0-255.
	int brightness = (int)(127.5f * (y_value + 1));

	// Set the brightness of the LED. If the specified pin does not support PWM,
	// this will result in the LED being on when y > 127, off otherwise.
	//  pwm.analogWrite(ledPin, brightness);

	// Log the current brightness value for display in the Arduino plotter
	TF_LITE_REPORT_ERROR(error_reporter, "%d\n", brightness);
}

void loop()
{
	// Calculate an x value to feed into the model. We compare the current
	// inference_count to the number of inferences per cycle to determine
	// our position within the range of possible x values the model was
	// trained on, and use this to calculate a value.
	float position = static_cast<float>(inference_count) / static_cast<float>(kInferencesPerCycle);
	float x_val = position * kXrange;

	// Place our calculated x value in the model's input tensor
	input->data.f[0] = x_val;

	// Run inference, and report any error
	TfLiteStatus invoke_status = interpreter->Invoke();
	if(invoke_status != kTfLiteOk) {
		TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed on x_val: %f\n", static_cast<double>(x_val));
		return;
	}

	// Read the predicted y value from the model's output tensor
	float y_val = output->data.f[0];

	// Output the results. A custom HandleOutput function can be implemented
	// for each supported hardware target.
	handleOutput(error_reporter, x_val, y_val);

	// Increment the inference_counter, and reset it if we have reached
	// the total number per cycle
	inference_count += 1;
	if(inference_count >= kInferencesPerCycle)
		inference_count = 0;
}

void init()
{
	// Set up logging. Google style is to avoid globals or statics because of
	// lifetime uncertainty, but since this has a trivial destructor it's okay.
	// NOLINTNEXTLINE(runtime-global-variables)
	static tflite::MicroErrorReporter micro_error_reporter;
	error_reporter = &micro_error_reporter;

	// Map the model into a usable data structure. This doesn't involve any
	// copying or parsing, it's a very lightweight operation.
	model = tflite::GetModel(g_sine_model_data);
	if(model->version() != TFLITE_SCHEMA_VERSION) {
		TF_LITE_REPORT_ERROR(error_reporter,
							 "Model provided is schema version %d not equal "
							 "to supported version %d.",
							 model->version(), TFLITE_SCHEMA_VERSION);
		return;
	}

	// This pulls in all the operation implementations we need.
	// NOLINTNEXTLINE(runtime-global-variables)
	static tflite::ops::micro::AllOpsResolver resolver;

	// Build an interpreter to run the model with.
	static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
	interpreter = &static_interpreter;

	// Allocate memory from the tensor_arena for the model's tensors.
	TfLiteStatus allocate_status = interpreter->AllocateTensors();
	if(allocate_status != kTfLiteOk) {
		TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
		return;
	}

	// Obtain pointers to the model's input and output tensors.
	input = interpreter->input(0);
	output = interpreter->output(0);

	// Keep track of how many inferences we have performed.
	inference_count = 0;

	// Start reading loop
	procTimer.initializeMs(100, loop).start();
}
