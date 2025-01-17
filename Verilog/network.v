module battery_soh_predictor (
    input wire clk,                 // Clock signal
    input wire reset,               // Reset signal
    input wire [127:0] in_data,     // (fixed-point, 16 fractional bits
    //flattened weights and biases
    input wire [64*4*32-1:0] weights1, 
    input wire [64*32-1:0] bias1,
    input wire [64*32*32-1:0] weights2,
    input wire [32*32-1:0] bias2,
    input wire [16*32*32-1:0] weights3,
    input wire [16*32-1:0] bias3,
    input wire [16*32-1:0] weights4,
    input wire [32-1:0] bias4,
    
    output reg [31:0] soh_out       // Output SoH prediction (fixed-point 
);
    // Parameters for layer sizes
    parameter INPUT_SIZE = 4;
    parameter LAYER1_SIZE = 64;
    parameter LAYER2_SIZE = 32;
    parameter LAYER3_SIZE = 16;
    parameter OUTPUT_SIZE = 1;

    // Internal wires between layers
    wire [32*64-1:0] layer1_out;
    wire [32*64-1:0] layer1_fin;
    wire [32*32-1:0] layer2_out;
    wire [32*32-1:0] layer2_fin;
    wire [32*16-1:0] layer3_out;
    wire [32*16-1:0] layer3_fin;
    wire [31:0] final_out;

    // Layer 1: Linear Layer + ReLU
    linear_layer #(
        .IN_SIZE(INPUT_SIZE),
        .OUT_SIZE(LAYER1_SIZE)
    ) layer1 (
        .clk(clk),
        .reset(reset),
        .in_data(in_data),
        .out_data(layer1_out),
        .weights(weights1),
        .biases(bias1)
    );

    relu_activation #(.SIZE(LAYER1_SIZE)) relu1 (
        .in_data(layer1_out),
        .out_data(layer1_fin)
    );

    // Layer 2: Linear Layer + ReLU
    linear_layer #(
        .IN_SIZE(LAYER1_SIZE),
        .OUT_SIZE(LAYER2_SIZE)
    ) layer2 (
        .clk(clk),
        .reset(reset),
        .in_data(layer1_fin),
        .out_data(layer2_out),
        .weights(weights2),
        .biases(bias2)
    );

    relu_activation #(.SIZE(LAYER2_SIZE)) relu2 (
        .in_data(layer2_out),
        .out_data(layer2_fin)
    );

    // Layer 3: Linear Layer + ReLU
    linear_layer #(
        .IN_SIZE(LAYER2_SIZE),
        .OUT_SIZE(LAYER3_SIZE)
    ) layer3 (
        .clk(clk),
        .reset(reset),
        .in_data(layer2_fin),
        .out_data(layer3_out),
        .weights(weights3),
        .biases(bias3)
    );

    relu_activation #(.SIZE(LAYER3_SIZE)) relu3 (
        .in_data(layer3_out),
        .out_data(layer3_fin)
    );

    // Final Layer: Linear Layer (Output)
    linear_layer #(
        .IN_SIZE(LAYER3_SIZE),
        .OUT_SIZE(OUTPUT_SIZE)
    ) final_layer (
        .clk(clk),
        .reset(reset),
        .in_data(layer3_fin),
        .out_data(final_out),
        .weights(weights4),
        .biases(bias4)
    );

    // Assign output
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            soh_out <= 32'b0;
        end else begin
            soh_out <= final_out;
        end
    end
endmodule

// Linear Layer Module
module linear_layer #(
    parameter IN_SIZE = 4,            // Input vector size
    parameter OUT_SIZE = 64          // Output vector size
)(
    input wire clk,
    input wire reset,
    input wire [IN_SIZE*32-1:0] in_data,          // Flattened input vector
    input wire [OUT_SIZE*IN_SIZE*32-1:0] weights, // Flattened weight matrix
    input wire [OUT_SIZE*32-1:0] biases,          // Biases
    output reg [OUT_SIZE*32-1:0] out_data         // Flattened output vector
);

    // Internal signals
    reg signed [31:0] in_vec[0:IN_SIZE-1];               // Unpacked input vector
    reg signed [31:0] weight_matrix[0:IN_SIZE-1][0:OUT_SIZE-1]; // Unpacked weight matrix
    reg signed [31:0] bias_vec[0:OUT_SIZE-1];            // Unpacked bias vector
    reg signed [31:0] temp_out[0:OUT_SIZE-1];            // Temporary output storage
    reg signed [63:0] temp_internal;            // Temporary output storage


    integer i, j;

    // Unpack the input data, weights, and biases
    always @(*) begin
        for (i = 0; i < IN_SIZE; i = i + 1) begin
            in_vec[i] = in_data[(IN_SIZE-i-1)*32 +: 32];
        end

        for (i = 0; i < OUT_SIZE; i = i + 1) begin
            for (j = 0; j < IN_SIZE; j = j + 1) begin
                weight_matrix[j][i] = weights[(IN_SIZE*OUT_SIZE-i*IN_SIZE - j -1)*32 +: 32];
            end
            bias_vec[i] = biases[(OUT_SIZE-i-1)*32 +: 32];
        end
    end

    // Perform matrix-vector multiplication and add bias
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            for (i = 0; i < OUT_SIZE; i = i + 1) begin
                temp_out[i] <= 32'b0;
                out_data[i*32 +: 32] <= 32'b0;
            end
        end else begin
            for (i = 0; i < OUT_SIZE; i = i + 1) begin
                temp_out[i] = bias_vec[i]; // Start with bias
                for (j = 0; j < IN_SIZE; j = j + 1) begin
                    temp_internal = in_vec[j] * weight_matrix[j][i];
                    temp_internal = temp_internal >>> 16;
                    temp_out[i] = temp_out[i] + temp_internal;
                end
                out_data[(OUT_SIZE-i-1)*32 +: 32] <= temp_out[i];
            end
        end
    end

endmodule




// ReLU Activation Module
module relu_activation #(
    parameter SIZE = 64
)(
    input wire [SIZE*32-1:0] in_data,
    output reg [SIZE*32-1:0] out_data
);
    integer i;
    reg signed [31:0] temp;

    always @(*) begin
        for (i = 0; i < SIZE; i = i + 1) begin
            temp = in_data[i*32 +: 32];
            out_data[i*32 +: 32] = (temp > 0) ? temp : 0;
        end
    end
endmodule
