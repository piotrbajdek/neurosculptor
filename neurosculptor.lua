#!/usr/bin/env lua

--[[

BSD 3-Clause License

Copyright © 2023-present, Piotr Bajdek
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

]]

local red = "\27[31m"
local grey = "\27[38;5;246m"
local reset = "\27[0m"

print("Lua-Based Deep Learning Program")
print(grey .. "neurosculptor" .. reset .. " v0.4.0 14.09.2023")
print("Copyright © 2023-present, Piotr Bajdek")
print("")

-- Otwieranie pliku do odczytu
local file = io.open("activation_input.conf", "r")
if file then
    -- Wczytanie jednego słowa ze strumienia
    activation_input = file:read("*a")
    -- Zamknięcie pliku
    file:close()
    -- Usunięcie białych znaków na początku i końcu słowa
    activation_input = activation_input:match("^%s*(.-)%s*$")
    -- Wyświetlenie wczytanego słowa
    print("Loaded " .. grey .. "activation_input.conf" .. reset .. ":" .. red, activation_input .. reset)
else
    -- Sprawdzenie, czy udało się wczytać activation_input
    print(red .. "Error when opening the file activation_input.conf" .. reset)
end

if activation_input == "colu" or activation_input == "CoLU" then
    function activation_input(x)
        local exp_x = math.exp(x)
        return x / (1 - x ^ -(x + exp_x))
    end
elseif activation_input == "gelu" or activation_input == "GELU" then
    function activation_input(x)
        return 0.5
            * x
            * (
                1
                + (math.exp(2 * (math.sqrt(2 / math.pi) * (x + 0.044715 * x * x * x))) - 1)
                    / (math.exp(2 * (math.sqrt(2 / math.pi) * (x + 0.044715 * x * x * x))) + 1)
            )
    end
elseif activation_input == "identity" or activation_input == "Identity" then
    function activation_input(x)
        return x
    end
elseif activation_input == "relu" or activation_input == "ReLU" then
    function activation_input(x)
        if x > 0 then
            return x
        else
            return 0
        end
    end
elseif activation_input == "sigmoid" or activation_input == "Sigmoid" then
    function activation_input(x)
        return 1 / (1 + math.exp(-x))
    end
elseif activation_input == "silu" or activation_input == "SiLU" then
    function activation_input(x)
        return x * (1 / (1 + math.exp(-x)))
    end
elseif activation_input == "swish" or activation_input == "Swish" then
    function activation_input(x)
        return x / (1 + math.exp(-x))
    end
elseif activation_input == "tanh" or activation_input == "Tanh" then
    function activation_input(x)
        return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
    end
elseif activation_input == "tanhexp" or activation_input == "TanhExp" then
    function activation_input(x)
        local exp_x = math.exp(x)
        local exp_neg_x = math.exp(-x)
        return x * (exp_x - exp_neg_x) / (exp_x + exp_neg_x)
    end
else
    print(red .. "Invalid value in activation_input.conf" .. reset)
end

-- Otwieranie pliku do odczytu
local file = io.open("activation_hidden.conf", "r")
if file then
    -- Wczytanie jednego słowa ze strumienia
    activation_hd = file:read("*a")
    -- Zamknięcie pliku
    file:close()
    -- Usunięcie białych znaków na początku i końcu słowa
    activation_hd = activation_hd:match("^%s*(.-)%s*$")
    -- Wyświetlenie wczytanego słowa
    print("Loaded " .. grey .. "activation_hidden.conf" .. reset .. ": " .. red, activation_hd .. reset)
else
    -- Sprawdzenie, czy udało się wczytać activation_hd
    print(red .. "Error when opening the file activation_hidden.conf" .. reset)
end

if activation_hd == "colu" or activation_hd == "CoLU" then
    function activation_hd(x)
        local exp_x = math.exp(x)
        return x / (1 - x ^ -(x + exp_x))
    end
elseif activation_hd == "gelu" or activation_hd == "GELU" then
    function activation_hd(x)
        return 0.5
            * x
            * (
                1
                + (math.exp(2 * (math.sqrt(2 / math.pi) * (x + 0.044715 * x * x * x))) - 1)
                    / (math.exp(2 * (math.sqrt(2 / math.pi) * (x + 0.044715 * x * x * x))) + 1)
            )
    end
elseif activation_hd == "identity" or activation_hd == "Identity" then
    function activation_hd(x)
        return x
    end
elseif activation_hd == "relu" or activation_hd == "ReLU" then
    function activation_hd(x)
        if x > 0 then
            return x
        else
            return 0
        end
    end
elseif activation_hd == "sigmoid" or activation_hd == "Sigmoid" then
    function activation_hd(x)
        return 1 / (1 + math.exp(-x))
    end
elseif activation_hd == "silu" or activation_hd == "SiLU" then
    function activation_hd(x)
        return x * (1 / (1 + math.exp(-x)))
    end
elseif activation_hd == "swish" or activation_hd == "Swish" then
    function activation_hd(x)
        return x / (1 + math.exp(-x))
    end
elseif activation_hd == "tanh" or activation_hd == "Tanh" then
    function activation_hd(x)
        return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
    end
elseif activation_hd == "tanhexp" or activation_hd == "TanhExp" then
    function activation_hd(x)
        local exp_x = math.exp(x)
        local exp_neg_x = math.exp(-x)
        return x * (exp_x - exp_neg_x) / (exp_x + exp_neg_x)
    end
else
    print(red .. "Invalid value in activation_hd.conf" .. reset)
end

-- Otwieranie pliku do odczytu
local file = io.open("activation_output.conf", "r")
if file then
    -- Wczytanie jednego słowa ze strumienia
    activation_output = file:read("*a")
    -- Zamknięcie pliku
    file:close()
    -- Usunięcie białych znaków na początku i końcu słowa
    activation_output = activation_output:match("^%s*(.-)%s*$")
    -- Wyświetlenie wczytanego słowa
    print("Loaded " .. grey .. "activation_output.conf" .. reset .. ":" .. red, activation_output .. reset)
else
    -- Sprawdzenie, czy udało się wczytać activation_output
    print(red .. "Error when opening the file activation_output.conf" .. reset)
end

if activation_output == "colu" or activation_output == "CoLU" then
    function activation_output(x)
        local exp_x = math.exp(x)
        return x / (1 - x ^ -(x + exp_x))
    end
elseif activation_output == "gelu" or activation_output == "GELU" then
    function activation_output(x)
        return 0.5
            * x
            * (
                1
                + (math.exp(2 * (math.sqrt(2 / math.pi) * (x + 0.044715 * x * x * x))) - 1)
                    / (math.exp(2 * (math.sqrt(2 / math.pi) * (x + 0.044715 * x * x * x))) + 1)
            )
    end
elseif activation_output == "identity" or activation_output == "Identity" then
    function activation_output(x)
        return x
    end
elseif activation_output == "relu" or activation_output == "ReLU" then
    function activation_output(x)
        if x > 0 then
            return x
        else
            return 0
        end
    end
elseif activation_output == "sigmoid" or activation_output == "Sigmoid" then
    function activation_output(x)
        return 1 / (1 + math.exp(-x))
    end
elseif activation_output == "silu" or activation_output == "SiLU" then
    function activation_output(x)
        return x * (1 / (1 + math.exp(-x)))
    end
elseif activation_output == "swish" or activation_output == "Swish" then
    function activation_output(x)
        return x / (1 + math.exp(-x))
    end
elseif activation_output == "tanh" or activation_output == "Tanh" then
    function activation_output(x)
        return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
    end
elseif activation_output == "tanhexp" or activation_output == "TanhExp" then
    function activation_output(x)
        local exp_x = math.exp(x)
        local exp_neg_x = math.exp(-x)
        return x * (exp_x - exp_neg_x) / (exp_x + exp_neg_x)
    end
else
    print(red .. "Invalid value in activation_output.conf" .. reset)
end

local file = io.open("optimisation.conf", "r")
if file then
    -- Wczytanie jednego słowa ze strumienia
    optimisation = file:read("*a")
    -- Zamknięcie pliku
    file:close()
    -- Usunięcie białych znaków na początku i końcu słowa
    optimisation = optimisation:match("^%s*(.-)%s*$")
    -- Wyświetlenie wczytanego słowa
    print("Loaded " .. grey .. "optimisation.conf" .. reset .. ":" .. red, optimisation .. reset)
else
    -- Sprawdzenie, czy udało się wczytać optimisation
    print(red .. "Error when opening the file optimisation.conf" .. reset)
end

if optimisation == "sgd" or optimisation == "SGD" then
    -- Otwieranie pliku do odczytu
    local file = io.open("learning_rate.conf", "r")
    -- Inicjalizacja zmiennej na learning_rate
    learning_rate = nil
    -- Sprawdzenie, czy plik został otwarty poprawnie
    if file then
        -- Odczytanie zawartości pliku
        local content = file:read("*all")
        -- Konwersja zawartości na liczbę (jeśli to możliwe)
        learning_rate = tonumber(content)
        file:close() -- Zamykanie pliku
    else
        print(red .. "Error when opening the file learning_rate.conf" .. reset)
    end
    -- Sprawdzenie, czy udało się wczytać learning_rate
    if learning_rate then
        print("Loaded " .. grey .. "learning_rate.conf" .. reset .. ":" .. red, learning_rate .. reset)
    else
        print(red .. "Unable to read learning_rate from learning_rate.conf" .. reset)
    end
end

if optimisation == "adam" or optimisation == "AdamW" or optimisation == "ADAMW" then
    -- Otwieranie pliku do odczytu
    local file = io.open("weight_decay.conf", "r")
    -- Inicjalizacja zmiennej na weight_decay
    weight_decay = nil
    -- Sprawdzenie, czy plik został otwarty poprawnie
    if file then
        -- Odczytanie zawartości pliku
        local content = file:read("*all")
        -- Konwersja zawartości na liczbę (jeśli to możliwe)
        weight_decay = tonumber(content)
        file:close() -- Zamykanie pliku
    else
        print(red .. "Error when opening the file weight_decay.conf" .. reset)
    end
    -- Sprawdzenie, czy udało się wczytać weight_decay
    if weight_decay then
        print("Loaded " .. grey .. "weight_decay.conf" .. reset .. ":" .. red, weight_decay .. reset)
    else
        print(red .. "Unable to read weight_decay from weight_decay.conf" .. reset)
    end
end

-- Otwieranie pliku do odczytu
local file = io.open("iterations.conf", "r")
-- Inicjalizacja zmiennej na iterations
local iterations = nil
-- Sprawdzenie, czy plik został otwarty poprawnie
if file then
    -- Odczytanie zawartości pliku
    local content = file:read("*all")
    -- Konwersja zawartości na liczbę (jeśli to możliwe)
    iterations = tonumber(content)
    file:close() -- Zamykanie pliku
else
    print(red .. "Error when opening the file iterations.conf" .. reset)
end
-- Sprawdzenie, czy udało się wczytać iterations
if iterations then
    print("Loaded " .. grey .. "iterations.conf" .. reset .. ":   " .. red, iterations .. reset)
else
    print(red .. "Unable to read iterations from iterations.conf" .. reset)
end

-- Otwieranie pliku do odczytu
local file = io.open("hidden_layers.conf", "r")
-- Inicjalizacja zmiennej na iterations
local hidden_layers = nil
-- Sprawdzenie, czy plik został otwarty poprawnie
if file then
    -- Odczytanie zawartości pliku
    local content = file:read("*all")
    -- Konwersja zawartości na liczbę (jeśli to możliwe)
    hidden_layers = tonumber(content)
    file:close() -- Zamykanie pliku
else
    print(red .. "Error when opening the file hidden_layers.conf" .. reset)
end
-- Sprawdzenie, czy udało się wczytać iterations
if hidden_layers then
    print("Loaded " .. grey .. "hidden_layers.conf" .. reset .. ":" .. red, hidden_layers .. reset)
else
    print(red .. "Unable to read hidden_layers from hidden_layers.conf" .. reset)
end
if hidden_layers <= 1 or hidden_layers >= 8 then
    print(red .. "Invalid value in hidden_layers.conf" .. reset)
end

if hidden_layers >= 1 then
    -- Otwieranie pliku do odczytu
    local file = io.open("hidden_1_size.conf", "r")
    -- Inicjalizacja zmiennej na iterations
    hidden1_size = nil
    -- Sprawdzenie, czy plik został otwarty poprawnie
    if file then
        -- Odczytanie zawartości pliku
        local content = file:read("*all")
        -- Konwersja zawartości na liczbę (jeśli to możliwe)
        hidden1_size = tonumber(content)
        file:close() -- Zamykanie pliku
    else
        print(red .. "Error when opening the file hidden_1_size.conf" .. reset)
    end
    -- Sprawdzenie, czy udało się wczytać iterations
    if hidden1_size then
        print("Loaded " .. grey .. "hidden_1_size.conf" .. reset .. ":" .. red, hidden1_size .. reset)
    else
        print(red .. "Unable to read hidden1_size from hidden_1_size.conf" .. reset)
    end
end

if hidden_layers >= 2 then
    -- Otwieranie pliku do odczytu
    local file = io.open("hidden_2_size.conf", "r")
    -- Inicjalizacja zmiennej na iterations
    hidden2_size = nil
    -- Sprawdzenie, czy plik został otwarty poprawnie
    if file then
        -- Odczytanie zawartości pliku
        local content = file:read("*all")
        -- Konwersja zawartości na liczbę (jeśli to możliwe)
        hidden2_size = tonumber(content)
        file:close() -- Zamykanie pliku
    else
        print(red .. "Error when opening the file hidden_2_size.conf" .. reset)
    end
    -- Sprawdzenie, czy udało się wczytać iterations
    if hidden2_size then
        print("Loaded " .. grey .. "hidden_2_size.conf" .. reset .. ":" .. red, hidden2_size .. reset)
    else
        print(red .. "Unable to read hidden2_size from hidden_2_size.conf" .. reset)
    end
end

if hidden_layers >= 3 then
    -- Otwieranie pliku do odczytu
    local file = io.open("hidden_3_size.conf", "r")
    -- Inicjalizacja zmiennej na iterations
    hidden3_size = nil
    -- Sprawdzenie, czy plik został otwarty poprawnie
    if file then
        -- Odczytanie zawartości pliku
        local content = file:read("*all")
        -- Konwersja zawartości na liczbę (jeśli to możliwe)
        hidden3_size = tonumber(content)
        file:close() -- Zamykanie pliku
    else
        print(red .. "Error when opening the file hidden_3_size.conf" .. reset)
    end
    -- Sprawdzenie, czy udało się wczytać iterations
    if hidden3_size then
        print("Loaded " .. grey .. "hidden_3_size.conf" .. reset .. ":" .. red, hidden3_size .. reset)
    else
        print(red .. "Unable to read hidden3_size from hidden_3_size.conf" .. reset)
    end
end

if hidden_layers >= 4 then
    -- Otwieranie pliku do odczytu
    local file = io.open("hidden_4_size.conf", "r")
    -- Inicjalizacja zmiennej na iterations
    hidden4_size = nil
    -- Sprawdzenie, czy plik został otwarty poprawnie
    if file then
        -- Odczytanie zawartości pliku
        local content = file:read("*all")
        -- Konwersja zawartości na liczbę (jeśli to możliwe)
        hidden4_size = tonumber(content)
        file:close() -- Zamykanie pliku
    else
        print(red .. "Error when opening the file hidden_4_size.conf" .. reset)
    end
    -- Sprawdzenie, czy udało się wczytać iterations
    if hidden4_size then
        print("Loaded " .. grey .. "hidden_4_size.conf" .. reset .. ":" .. red, hidden4_size .. reset)
    else
        print(red .. "Unable to read hidden4_size from hidden_4_size.conf" .. reset)
    end
end

if hidden_layers >= 5 then
    -- Otwieranie pliku do odczytu
    local file = io.open("hidden_5_size.conf", "r")
    -- Inicjalizacja zmiennej na iterations
    hidden5_size = nil
    -- Sprawdzenie, czy plik został otwarty poprawnie
    if file then
        -- Odczytanie zawartości pliku
        local content = file:read("*all")
        -- Konwersja zawartości na liczbę (jeśli to możliwe)
        hidden5_size = tonumber(content)
        file:close() -- Zamykanie pliku
    else
        print(red .. "Error when opening the file hidden_5_size.conf" .. reset)
    end
    -- Sprawdzenie, czy udało się wczytać iterations
    if hidden5_size then
        print("Loaded " .. grey .. "hidden_5_size.conf" .. reset .. ":" .. red, hidden5_size .. reset)
    else
        print(red .. "Unable to read hidden5_size from hidden_5_size.conf" .. reset)
    end
end

if hidden_layers >= 6 then
    -- Otwieranie pliku do odczytu
    local file = io.open("hidden_6_size.conf", "r")
    -- Inicjalizacja zmiennej na iterations
    hidden6_size = nil
    -- Sprawdzenie, czy plik został otwarty poprawnie
    if file then
        -- Odczytanie zawartości pliku
        local content = file:read("*all")
        -- Konwersja zawartości na liczbę (jeśli to możliwe)
        hidden6_size = tonumber(content)
        file:close() -- Zamykanie pliku
    else
        print(red .. "Error when opening the file hidden_6_size.conf" .. reset)
    end
    -- Sprawdzenie, czy udało się wczytać iterations
    if hidden6_size then
        print("Loaded " .. grey .. "hidden_6_size.conf" .. reset .. ":" .. red, hidden6_size .. reset)
    else
        print(red .. "Unable to read hidden6_size from hidden_6_size.conf" .. reset)
    end
end

if hidden_layers >= 7 then
    -- Otwieranie pliku do odczytu
    local file = io.open("hidden_7_size.conf", "r")
    -- Inicjalizacja zmiennej na iterations
    hidden7_size = nil
    -- Sprawdzenie, czy plik został otwarty poprawnie
    if file then
        -- Odczytanie zawartości pliku
        local content = file:read("*all")
        -- Konwersja zawartości na liczbę (jeśli to możliwe)
        hidden7_size = tonumber(content)
        file:close() -- Zamykanie pliku
    else
        print(red .. "Error when opening the file hidden_7_size.conf" .. reset)
    end
    -- Sprawdzenie, czy udało się wczytać iterations
    if hidden7_size then
        print("Loaded " .. grey .. "hidden_7_size.conf" .. reset .. ":" .. red, hidden7_size .. reset)
    else
        print(red .. "Unable to read hidden7_size from hidden_7_size.conf" .. reset)
    end
end

-- Otwieranie pliku do odczytu
local file = io.open("train_file_x.csv", "r")
-- Inicjalizacja tablicy na dane testowe
local x_train = {}
-- Sprawdzenie, czy plik został otwarty poprawnie
if file then
    for line in file:lines() do
        local row = {}
        for value in line:gmatch("[^,]+") do
            -- Parsowanie wartości jako liczby
            table.insert(row, tonumber(value))
        end
        table.insert(x_train, row)
    end
    file:close() -- Zamykanie pliku
    print("Loaded " .. grey .. "train_file_x.csv" .. reset)
else
    print(red .. "Error when opening the file train_file_x.csv" .. reset)
end

-- Otwieranie pliku do odczytu
local file = io.open("train_file_y.txt", "r")
-- Inicjalizacja tabeli y_train
local y_train = {}
-- Sprawdzenie, czy plik został otwarty poprawnie
if file then
    -- Odczytanie zawartości pliku linia po linii
    for line in file:lines() do
        -- Konwersja zawartości linii na liczbę i dodanie jej do tabeli y_train
        local number = tonumber(line)
        if number then
            table.insert(y_train, number)
        else
            print(red .. "Error while converting a line to a number:", line .. reset)
        end
    end
    file:close() -- Zamykanie pliku
else
    print(red .. "Error when opening the file train_file_y.txt" .. reset)
end
-- Sprawdzenie, czy udało się wczytać y_train
if #y_train > 0 then
    print("Loaded " .. grey .. "train_file_y.txt" .. reset)
else
    print(red .. "Unable to read y_train from train_file_y.txt" .. reset)
end

-- Otwieranie pliku do odczytu
local file = io.open("test_file.csv", "r")
-- Inicjalizacja tablicy na dane testowe
local x_test = {}
-- Sprawdzenie, czy plik został otwarty poprawnie
if file then
    for line in file:lines() do
        local row = {}
        for value in line:gmatch("[^,]+") do
            -- Parsowanie wartości jako liczby
            table.insert(row, tonumber(value))
        end
        table.insert(x_test, row)
    end
    file:close() -- Zamykanie pliku
    print("Loaded " .. grey .. "test_file.csv" .. reset)
else
    print(red .. "Error when opening the file test_file.csv" .. reset)
end

-- Otwieranie pliku do odczytu
local file = io.open("train_file_x.csv", "r")
-- Inicjalizacja zmiennej na input_size
local input_size = nil
-- Sprawdzenie, czy plik został otwarty poprawnie
if file then
    -- Odczytanie pierwszej linii (nagłówka) pliku
    local header = file:read()
    -- Podzielenie nagłówka na kolumny, używając przecinka jako separatora
    local columns = {}
    for column in header:gmatch("[^,]+") do
        table.insert(columns, column)
    end
    -- Obliczenie input_size jako liczby kolumn
    input_size = #columns
    file:close() -- Zamykanie pliku
else
    print(red .. "Error when opening the file train_file_x.csv" .. reset)
end
-- Sprawdzenie, czy udało się obliczyć input_size
if input_size then
    print("")
    print("Configured input neurons:   " .. red, input_size .. reset)
else
    print(red .. "Failed to compute input_size" .. reset)
end

output_size = 1
if hidden_layers == 7 then
    neurons = input_size
        + hidden1_size
        + hidden2_size
        + hidden3_size
        + hidden4_size
        + hidden5_size
        + hidden6_size
        + hidden7_size
        + output_size
    print("Configured hidden neurons:" .. red, neurons - input_size - output_size .. reset)
elseif hidden_layers == 6 then
    neurons = input_size
        + hidden1_size
        + hidden2_size
        + hidden3_size
        + hidden4_size
        + hidden5_size
        + hidden6_size
        + output_size
    print("Configured hidden neurons:" .. red, neurons - input_size - output_size .. reset)
elseif hidden_layers == 5 then
    neurons = input_size + hidden1_size + hidden2_size + hidden3_size + hidden4_size + hidden5_size + output_size
    print("Configured hidden neurons:" .. red, neurons - input_size - output_size .. reset)
elseif hidden_layers == 4 then
    neurons = input_size + hidden1_size + hidden2_size + hidden3_size + hidden4_size + output_size
    print("Configured hidden neurons:" .. red, neurons - input_size - output_size .. reset)
elseif hidden_layers == 3 then
    neurons = input_size + hidden1_size + hidden2_size + hidden3_size + output_size
    print("Configured hidden neurons:" .. red, neurons - input_size - output_size .. reset)
elseif hidden_layers == 2 then
    neurons = input_size + hidden1_size + hidden2_size + output_size
    print("Configured hidden neurons:" .. red, neurons - input_size - output_size .. reset)
end
print("Configured output neurons:   " .. red, output_size .. reset)
print("Configured total layers:" .. red, hidden_layers + 2 .. reset)
print("Configured total neurons:" .. red, neurons .. reset)

if hidden_layers == 7 then
    parameters = (input_size * hidden1_size)
        + (hidden1_size * hidden2_size)
        + (hidden2_size * hidden3_size)
        + (hidden3_size * hidden4_size)
        + (hidden4_size * hidden5_size)
        + (hidden5_size * hidden6_size)
        + (hidden6_size * hidden7_size)
        + (hidden7_size * output_size)
    print("Configured total connections:" .. red, parameters .. reset)
elseif hidden_layers == 6 then
    parameters = (input_size * hidden1_size)
        + (hidden1_size * hidden2_size)
        + (hidden2_size * hidden3_size)
        + (hidden3_size * hidden4_size)
        + (hidden4_size * hidden5_size)
        + (hidden5_size * hidden6_size)
        + (hidden6_size * output_size)
    print("Configured total connections:" .. red, parameters .. reset)
elseif hidden_layers == 5 then
    parameters = (input_size * hidden1_size)
        + (hidden1_size * hidden2_size)
        + (hidden2_size * hidden3_size)
        + (hidden3_size * hidden4_size)
        + (hidden4_size * hidden5_size)
        + (hidden5_size * output_size)
    print("Configured total connections:" .. red, parameters .. reset)
elseif hidden_layers == 4 then
    parameters = (input_size * hidden1_size)
        + (hidden1_size * hidden2_size)
        + (hidden2_size * hidden3_size)
        + (hidden3_size * hidden4_size)
        + (hidden4_size * output_size)
    print("Configured total connections:" .. red, parameters .. reset)
elseif hidden_layers == 3 then
    parameters = (input_size * hidden1_size)
        + (hidden1_size * hidden2_size)
        + (hidden2_size * hidden3_size)
        + (hidden3_size * output_size)
    print("Configured total connections:" .. red, parameters .. reset)
elseif hidden_layers == 2 then
    parameters = (input_size * hidden1_size) + (hidden1_size * hidden2_size) + (hidden2_size * output_size)
    print("Configured total connections:" .. red, parameters .. reset)
end

if hidden_layers == 7 then
    parameters = (input_size * hidden1_size + hidden1_size)
        + (hidden1_size * hidden2_size + hidden2_size)
        + (hidden2_size * hidden3_size + hidden3_size)
        + (hidden3_size * hidden4_size + hidden4_size)
        + (hidden4_size * hidden5_size + hidden5_size)
        + (hidden5_size * hidden6_size + hidden6_size)
        + (hidden6_size * hidden7_size + hidden7_size)
        + (hidden7_size * output_size + output_size)
    print("Configured total parameters:" .. red, parameters .. reset)
elseif hidden_layers == 6 then
    parameters = (input_size * hidden1_size + hidden1_size)
        + (hidden1_size * hidden2_size + hidden2_size)
        + (hidden2_size * hidden3_size + hidden3_size)
        + (hidden3_size * hidden4_size + hidden4_size)
        + (hidden4_size * hidden5_size + hidden5_size)
        + (hidden5_size * hidden6_size + hidden6_size)
        + (hidden6_size * output_size + output_size)
    print("Configured total parameters:" .. red, parameters .. reset)
elseif hidden_layers == 5 then
    parameters = (input_size * hidden1_size + hidden1_size)
        + (hidden1_size * hidden2_size + hidden2_size)
        + (hidden2_size * hidden3_size + hidden3_size)
        + (hidden3_size * hidden4_size + hidden4_size)
        + (hidden4_size * hidden5_size + hidden5_size)
        + (hidden5_size * output_size + output_size)
    print("Configured total parameters:" .. red, parameters .. reset)
elseif hidden_layers == 4 then
    parameters = (input_size * hidden1_size + hidden1_size)
        + (hidden1_size * hidden2_size + hidden2_size)
        + (hidden2_size * hidden3_size + hidden3_size)
        + (hidden3_size * hidden4_size + hidden4_size)
        + (hidden4_size * output_size + output_size)
    print("Configured total parameters:" .. red, parameters .. reset)
elseif hidden_layers == 3 then
    parameters = (input_size * hidden1_size + hidden1_size)
        + (hidden1_size * hidden2_size + hidden2_size)
        + (hidden2_size * hidden3_size + hidden3_size)
        + (hidden3_size * output_size + output_size)
    print("Configured total parameters:" .. red, parameters .. reset)
elseif hidden_layers == 2 then
    parameters = (input_size * hidden1_size + hidden1_size)
        + (hidden1_size * hidden2_size + hidden2_size)
        + (hidden2_size * output_size + output_size)
    print("Configured total parameters:" .. red, parameters .. reset)
end

-- Inicjalizacja wag
math.randomseed(os.time())

if optimisation == "sgd" or optimisation == "SGD" then
    if hidden_layers == 2 then
        weights_input_hidden1 = {}
        weights_hidden1_hidden2 = {}
        weights_hidden2_output = {}

        for i = 1, input_size do
            weights_input_hidden1[i] = {}
            for j = 1, hidden1_size do
                weights_input_hidden1[i][j] = math.random()
            end
        end

        for i = 1, hidden1_size do
            weights_hidden1_hidden2[i] = {}
            for j = 1, hidden2_size do
                weights_hidden1_hidden2[i][j] = math.random()
            end
        end

        for i = 1, hidden2_size do
            weights_hidden2_output[i] = math.random()
        end

        -- Uczenie sieci

        epochs = iterations

        for epoch = 1, epochs do
            for i = 1, #x_train do
                -- Forward pass
                hidden1_input = {}
                hidden2_input = {}
                hidden2_output = {}
                final_input = {}

                for j = 1, hidden1_size do
                    hidden1_input[j] = 0
                    for k = 1, input_size do
                        hidden1_input[j] = hidden1_input[j] + x_train[i][k] * weights_input_hidden1[k][j]
                    end
                    hidden1_input[j] = activation_input(hidden1_input[j])
                end

                for j = 1, hidden2_size do
                    hidden2_input[j] = 0
                    for k = 1, hidden1_size do
                        hidden2_input[j] = hidden2_input[j] + hidden1_input[k] * weights_hidden1_hidden2[k][j]
                    end
                    hidden2_input[j] = activation_hd(hidden2_input[j])
                end

                for j = 1, hidden2_size do
                    hidden2_output[j] = activation_hd(hidden2_input[j])
                end

                final_input[1] = 0
                for j = 1, hidden2_size do
                    final_input[1] = final_input[1] + hidden2_output[j] * weights_hidden2_output[j]
                end
                final_output = activation_output(final_input[1])

                -- Obliczenie błędu
                error = y_train[i] - final_output

                -- Backpropagation
                delta_output = error * final_output * (1 - final_output)

                delta_hidden2 = {}
                for j = 1, hidden2_size do
                    delta_hidden2[j] = delta_output * weights_hidden2_output[j] * (1 - hidden2_output[j])
                end

                delta_hidden1 = {}
                for j = 1, hidden1_size do
                    delta_hidden1[j] = 0
                    for k = 1, hidden2_size do
                        delta_hidden1[j] = delta_hidden1[j]
                            + delta_hidden2[k] * weights_hidden1_hidden2[j][k] * (1 - hidden1_input[j])
                    end
                end

                -- Aktualizacja wag
                for j = 1, hidden2_size do
                    weights_hidden2_output[j] = weights_hidden2_output[j]
                        + learning_rate * delta_output * hidden2_output[j]
                end

                for j = 1, hidden1_size do
                    for k = 1, hidden2_size do
                        weights_hidden1_hidden2[j][k] = weights_hidden1_hidden2[j][k]
                            + learning_rate * delta_hidden2[k] * hidden1_input[j]
                    end
                end

                for j = 1, input_size do
                    for k = 1, hidden1_size do
                        weights_input_hidden1[j][k] = weights_input_hidden1[j][k]
                            + learning_rate * delta_hidden1[k] * x_train[i][j]
                    end
                end
            end
        end

        -- Testowanie sieci

        for i = 1, #x_test do
            print("")

            hidden1_input = {}
            hidden2_input = {}
            hidden2_output = {}
            final_input = {}

            for j = 1, hidden1_size do
                hidden1_input[j] = 0
                for k = 1, input_size do
                    hidden1_input[j] = hidden1_input[j] + x_test[i][k] * weights_input_hidden1[k][j]
                end
                hidden1_input[j] = activation_input(hidden1_input[j])
            end

            for j = 1, hidden2_size do
                hidden2_input[j] = 0
                for k = 1, hidden1_size do
                    hidden2_input[j] = hidden2_input[j] + hidden1_input[k] * weights_hidden1_hidden2[k][j]
                end
                hidden2_input[j] = activation_hd(hidden2_input[j])
            end

            for j = 1, hidden2_size do
                hidden2_output[j] = activation_hd(hidden2_input[j])
            end

            final_input[1] = 0
            for j = 1, hidden2_size do
                final_input[1] = final_input[1] + hidden2_output[j] * weights_hidden2_output[j]
            end
            final_output = activation_output(final_input[1])

            print("Test input:      " .. grey, table.concat(x_test[i], " ") .. reset)
            print("Predicted output:" .. red, final_output .. reset)
        end
    end

    if hidden_layers == 3 then
        weights_input_hidden1 = {}
        weights_hidden1_hidden2 = {}
        weights_hidden2_hidden3 = {}
        weights_hidden3_output = {}

        for i = 1, input_size do
            weights_input_hidden1[i] = {}
            for j = 1, hidden1_size do
                weights_input_hidden1[i][j] = math.random()
            end
        end

        for i = 1, hidden1_size do
            weights_hidden1_hidden2[i] = {}
            for j = 1, hidden2_size do
                weights_hidden1_hidden2[i][j] = math.random()
            end
        end

        for i = 1, hidden2_size do
            weights_hidden2_hidden3[i] = {}
            for j = 1, hidden3_size do
                weights_hidden2_hidden3[i][j] = math.random()
            end
        end

        for i = 1, hidden3_size do
            weights_hidden3_output[i] = math.random()
        end

        -- Uczenie sieci

        epochs = iterations

        for epoch = 1, epochs do
            for i = 1, #x_train do
                -- Forward pass
                hidden1_input = {}
                hidden2_input = {}
                hidden3_input = {}
                hidden3_output = {}
                final_input = {}

                for j = 1, hidden1_size do
                    hidden1_input[j] = 0
                    for k = 1, input_size do
                        hidden1_input[j] = hidden1_input[j] + x_train[i][k] * weights_input_hidden1[k][j]
                    end
                    hidden1_input[j] = activation_input(hidden1_input[j])
                end

                for j = 1, hidden2_size do
                    hidden2_input[j] = 0
                    for k = 1, hidden1_size do
                        hidden2_input[j] = hidden2_input[j] + hidden1_input[k] * weights_hidden1_hidden2[k][j]
                    end
                    hidden2_input[j] = activation_hd(hidden2_input[j])
                end

                for j = 1, hidden3_size do
                    hidden3_input[j] = 0
                    for k = 1, hidden2_size do
                        hidden3_input[j] = hidden3_input[j] + hidden2_input[k] * weights_hidden2_hidden3[k][j]
                    end
                    hidden3_input[j] = activation_hd(hidden3_input[j])
                end

                for j = 1, hidden3_size do
                    hidden3_output[j] = activation_hd(hidden3_input[j])
                end

                final_input[1] = 0
                for j = 1, hidden3_size do
                    final_input[1] = final_input[1] + hidden3_output[j] * weights_hidden3_output[j]
                end
                final_output = activation_output(final_input[1])

                -- Obliczenie błędu
                error = y_train[i] - final_output

                -- Backpropagation
                delta_output = error * final_output * (1 - final_output)

                delta_hidden3 = {}
                for j = 1, hidden3_size do
                    delta_hidden3[j] = delta_output * weights_hidden3_output[j] * (1 - hidden3_output[j])
                end

                delta_hidden2 = {}
                for j = 1, hidden2_size do
                    delta_hidden2[j] = 0
                    for k = 1, hidden3_size do
                        delta_hidden2[j] = delta_hidden2[j]
                            + delta_hidden3[k] * weights_hidden2_hidden3[j][k] * (1 - hidden2_input[j])
                    end
                end

                delta_hidden1 = {}
                for j = 1, hidden1_size do
                    delta_hidden1[j] = 0
                    for k = 1, hidden2_size do
                        delta_hidden1[j] = delta_hidden1[j]
                            + delta_hidden2[k] * weights_hidden1_hidden2[j][k] * (1 - hidden1_input[j])
                    end
                end

                -- Aktualizacja wag
                for j = 1, hidden3_size do
                    weights_hidden3_output[j] = weights_hidden3_output[j]
                        + learning_rate * delta_output * hidden3_output[j]
                end

                for j = 1, hidden2_size do
                    for k = 1, hidden3_size do
                        weights_hidden2_hidden3[j][k] = weights_hidden2_hidden3[j][k]
                            + learning_rate * delta_hidden3[k] * hidden2_input[j]
                    end
                end

                for j = 1, hidden1_size do
                    for k = 1, hidden2_size do
                        weights_hidden1_hidden2[j][k] = weights_hidden1_hidden2[j][k]
                            + learning_rate * delta_hidden2[k] * hidden1_input[j]
                    end
                end

                for j = 1, input_size do
                    for k = 1, hidden1_size do
                        weights_input_hidden1[j][k] = weights_input_hidden1[j][k]
                            + learning_rate * delta_hidden1[k] * x_train[i][j]
                    end
                end
            end
        end

        -- Testowanie sieci

        for i = 1, #x_test do
            print("")

            hidden1_input = {}
            hidden2_input = {}
            hidden3_input = {}
            hidden3_output = {}
            final_input = {}

            for j = 1, hidden1_size do
                hidden1_input[j] = 0
                for k = 1, input_size do
                    hidden1_input[j] = hidden1_input[j] + x_test[i][k] * weights_input_hidden1[k][j]
                end
                hidden1_input[j] = activation_input(hidden1_input[j])
            end

            for j = 1, hidden2_size do
                hidden2_input[j] = 0
                for k = 1, hidden1_size do
                    hidden2_input[j] = hidden2_input[j] + hidden1_input[k] * weights_hidden1_hidden2[k][j]
                end
                hidden2_input[j] = activation_hd(hidden2_input[j])
            end

            for j = 1, hidden3_size do
                hidden3_input[j] = 0
                for k = 1, hidden2_size do
                    hidden3_input[j] = hidden3_input[j] + hidden2_input[k] * weights_hidden2_hidden3[k][j]
                end
                hidden3_input[j] = activation_hd(hidden3_input[j])
            end

            for j = 1, hidden3_size do
                hidden3_output[j] = activation_hd(hidden3_input[j])
            end

            final_input[1] = 0
            for j = 1, hidden3_size do
                final_input[1] = final_input[1] + hidden3_output[j] * weights_hidden3_output[j]
            end
            final_output = activation_output(final_input[1])

            print("Test input:      " .. grey, table.concat(x_test[i], " ") .. reset)
            print("Predicted output:" .. red, final_output .. reset)
        end
    end

    if hidden_layers == 4 then
        weights_input_hidden1 = {}
        weights_hidden1_hidden2 = {}
        weights_hidden2_hidden3 = {}
        weights_hidden3_hidden4 = {}
        weights_hidden4_output = {}

        for i = 1, input_size do
            weights_input_hidden1[i] = {}
            for j = 1, hidden1_size do
                weights_input_hidden1[i][j] = math.random()
            end
        end

        for i = 1, hidden1_size do
            weights_hidden1_hidden2[i] = {}
            for j = 1, hidden2_size do
                weights_hidden1_hidden2[i][j] = math.random()
            end
        end

        for i = 1, hidden2_size do
            weights_hidden2_hidden3[i] = {}
            for j = 1, hidden3_size do
                weights_hidden2_hidden3[i][j] = math.random()
            end
        end

        for i = 1, hidden3_size do
            weights_hidden3_hidden4[i] = {}
            for j = 1, hidden4_size do
                weights_hidden3_hidden4[i][j] = math.random()
            end
        end

        for i = 1, hidden4_size do
            weights_hidden4_output[i] = math.random()
        end

        -- Uczenie sieci

        epochs = iterations

        for epoch = 1, epochs do
            for i = 1, #x_train do
                -- Forward pass
                hidden1_input = {}
                hidden2_input = {}
                hidden3_input = {}
                hidden4_input = {}
                hidden4_output = {}
                final_input = {}

                for j = 1, hidden1_size do
                    hidden1_input[j] = 0
                    for k = 1, input_size do
                        hidden1_input[j] = hidden1_input[j] + x_train[i][k] * weights_input_hidden1[k][j]
                    end
                    hidden1_input[j] = activation_input(hidden1_input[j])
                end

                for j = 1, hidden2_size do
                    hidden2_input[j] = 0
                    for k = 1, hidden1_size do
                        hidden2_input[j] = hidden2_input[j] + hidden1_input[k] * weights_hidden1_hidden2[k][j]
                    end
                    hidden2_input[j] = activation_hd(hidden2_input[j])
                end

                for j = 1, hidden3_size do
                    hidden3_input[j] = 0
                    for k = 1, hidden2_size do
                        hidden3_input[j] = hidden3_input[j] + hidden2_input[k] * weights_hidden2_hidden3[k][j]
                    end
                    hidden3_input[j] = activation_hd(hidden3_input[j])
                end

                for j = 1, hidden4_size do
                    hidden4_input[j] = 0
                    for k = 1, hidden3_size do
                        hidden4_input[j] = hidden4_input[j] + hidden3_input[k] * weights_hidden3_hidden4[k][j]
                    end
                    hidden4_input[j] = activation_hd(hidden4_input[j])
                end

                for j = 1, hidden4_size do
                    hidden4_output[j] = activation_hd(hidden4_input[j])
                end

                final_input[1] = 0
                for j = 1, hidden4_size do
                    final_input[1] = final_input[1] + hidden4_output[j] * weights_hidden4_output[j]
                end
                final_output = activation_output(final_input[1])

                -- Obliczenie błędu
                error = y_train[i] - final_output

                -- Backpropagation
                delta_output = error * final_output * (1 - final_output)

                delta_hidden4 = {}
                for j = 1, hidden4_size do
                    delta_hidden4[j] = delta_output * weights_hidden4_output[j] * (1 - hidden4_output[j])
                end

                delta_hidden3 = {}
                for j = 1, hidden3_size do
                    delta_hidden3[j] = 0
                    for k = 1, hidden4_size do
                        delta_hidden3[j] = delta_hidden3[j]
                            + delta_hidden4[k] * weights_hidden3_hidden4[j][k] * (1 - hidden3_input[j])
                    end
                end

                delta_hidden2 = {}
                for j = 1, hidden2_size do
                    delta_hidden2[j] = 0
                    for k = 1, hidden3_size do
                        delta_hidden2[j] = delta_hidden2[j]
                            + delta_hidden3[k] * weights_hidden2_hidden3[j][k] * (1 - hidden2_input[j])
                    end
                end

                delta_hidden1 = {}
                for j = 1, hidden1_size do
                    delta_hidden1[j] = 0
                    for k = 1, hidden2_size do
                        delta_hidden1[j] = delta_hidden1[j]
                            + delta_hidden2[k] * weights_hidden1_hidden2[j][k] * (1 - hidden1_input[j])
                    end
                end

                -- Aktualizacja wag
                for j = 1, hidden4_size do
                    weights_hidden4_output[j] = weights_hidden4_output[j]
                        + learning_rate * delta_output * hidden4_output[j]
                end

                for j = 1, hidden3_size do
                    for k = 1, hidden4_size do
                        weights_hidden3_hidden4[j][k] = weights_hidden3_hidden4[j][k]
                            + learning_rate * delta_hidden4[k] * hidden3_input[j]
                    end
                end

                for j = 1, hidden2_size do
                    for k = 1, hidden3_size do
                        weights_hidden2_hidden3[j][k] = weights_hidden2_hidden3[j][k]
                            + learning_rate * delta_hidden3[k] * hidden2_input[j]
                    end
                end

                for j = 1, hidden1_size do
                    for k = 1, hidden2_size do
                        weights_hidden1_hidden2[j][k] = weights_hidden1_hidden2[j][k]
                            + learning_rate * delta_hidden2[k] * hidden1_input[j]
                    end
                end

                for j = 1, input_size do
                    for k = 1, hidden1_size do
                        weights_input_hidden1[j][k] = weights_input_hidden1[j][k]
                            + learning_rate * delta_hidden1[k] * x_train[i][j]
                    end
                end
            end
        end

        -- Testowanie sieci

        for i = 1, #x_test do
            print("")

            hidden1_input = {}
            hidden2_input = {}
            hidden3_input = {}
            hidden4_input = {}
            hidden4_output = {}
            final_input = {}

            for j = 1, hidden1_size do
                hidden1_input[j] = 0
                for k = 1, input_size do
                    hidden1_input[j] = hidden1_input[j] + x_test[i][k] * weights_input_hidden1[k][j]
                end
                hidden1_input[j] = activation_input(hidden1_input[j])
            end

            for j = 1, hidden2_size do
                hidden2_input[j] = 0
                for k = 1, hidden1_size do
                    hidden2_input[j] = hidden2_input[j] + hidden1_input[k] * weights_hidden1_hidden2[k][j]
                end
                hidden2_input[j] = activation_hd(hidden2_input[j])
            end

            for j = 1, hidden3_size do
                hidden3_input[j] = 0
                for k = 1, hidden2_size do
                    hidden3_input[j] = hidden3_input[j] + hidden2_input[k] * weights_hidden2_hidden3[k][j]
                end
                hidden3_input[j] = activation_hd(hidden3_input[j])
            end

            for j = 1, hidden4_size do
                hidden4_input[j] = 0
                for k = 1, hidden3_size do
                    hidden4_input[j] = hidden4_input[j] + hidden3_input[k] * weights_hidden3_hidden4[k][j]
                end
                hidden4_input[j] = activation_hd(hidden4_input[j])
            end

            for j = 1, hidden4_size do
                hidden4_output[j] = activation_hd(hidden4_input[j])
            end

            final_input[1] = 0
            for j = 1, hidden4_size do
                final_input[1] = final_input[1] + hidden4_output[j] * weights_hidden4_output[j]
            end
            final_output = activation_output(final_input[1])

            print("Test input:      " .. grey, table.concat(x_test[i], " ") .. reset)
            print("Predicted output:" .. red, final_output .. reset)
        end
    end

    if hidden_layers == 5 then
        weights_input_hidden1 = {}
        weights_hidden1_hidden2 = {}
        weights_hidden2_hidden3 = {}
        weights_hidden3_hidden4 = {}
        weights_hidden4_hidden5 = {}
        weights_hidden5_output = {}

        for i = 1, input_size do
            weights_input_hidden1[i] = {}
            for j = 1, hidden1_size do
                weights_input_hidden1[i][j] = math.random()
            end
        end

        for i = 1, hidden1_size do
            weights_hidden1_hidden2[i] = {}
            for j = 1, hidden2_size do
                weights_hidden1_hidden2[i][j] = math.random()
            end
        end

        for i = 1, hidden2_size do
            weights_hidden2_hidden3[i] = {}
            for j = 1, hidden3_size do
                weights_hidden2_hidden3[i][j] = math.random()
            end
        end

        for i = 1, hidden3_size do
            weights_hidden3_hidden4[i] = {}
            for j = 1, hidden4_size do
                weights_hidden3_hidden4[i][j] = math.random()
            end
        end

        for i = 1, hidden4_size do
            weights_hidden4_hidden5[i] = {}
            for j = 1, hidden5_size do
                weights_hidden4_hidden5[i][j] = math.random()
            end
        end

        for i = 1, hidden5_size do
            weights_hidden5_output[i] = math.random()
        end

        -- Uczenie sieci

        epochs = iterations

        for epoch = 1, epochs do
            for i = 1, #x_train do
                -- Forward pass
                hidden1_input = {}
                hidden2_input = {}
                hidden3_input = {}
                hidden4_input = {}
                hidden5_input = {}
                hidden5_output = {}
                final_input = {}

                for j = 1, hidden1_size do
                    hidden1_input[j] = 0
                    for k = 1, input_size do
                        hidden1_input[j] = hidden1_input[j] + x_train[i][k] * weights_input_hidden1[k][j]
                    end
                    hidden1_input[j] = activation_input(hidden1_input[j])
                end

                for j = 1, hidden2_size do
                    hidden2_input[j] = 0
                    for k = 1, hidden1_size do
                        hidden2_input[j] = hidden2_input[j] + hidden1_input[k] * weights_hidden1_hidden2[k][j]
                    end
                    hidden2_input[j] = activation_hd(hidden2_input[j])
                end

                for j = 1, hidden3_size do
                    hidden3_input[j] = 0
                    for k = 1, hidden2_size do
                        hidden3_input[j] = hidden3_input[j] + hidden2_input[k] * weights_hidden2_hidden3[k][j]
                    end
                    hidden3_input[j] = activation_hd(hidden3_input[j])
                end

                for j = 1, hidden4_size do
                    hidden4_input[j] = 0
                    for k = 1, hidden3_size do
                        hidden4_input[j] = hidden4_input[j] + hidden3_input[k] * weights_hidden3_hidden4[k][j]
                    end
                    hidden4_input[j] = activation_hd(hidden4_input[j])
                end

                for j = 1, hidden5_size do
                    hidden5_input[j] = 0
                    for k = 1, hidden4_size do
                        hidden5_input[j] = hidden5_input[j] + hidden4_input[k] * weights_hidden4_hidden5[k][j]
                    end
                    hidden5_input[j] = activation_hd(hidden5_input[j])
                end

                for j = 1, hidden5_size do
                    hidden5_output[j] = activation_hd(hidden5_input[j])
                end

                final_input[1] = 0
                for j = 1, hidden5_size do
                    final_input[1] = final_input[1] + hidden5_output[j] * weights_hidden5_output[j]
                end
                final_output = activation_output(final_input[1])

                -- Obliczenie błędu
                error = y_train[i] - final_output

                -- Backpropagation
                delta_output = error * final_output * (1 - final_output)

                delta_hidden5 = {}
                for j = 1, hidden5_size do
                    delta_hidden5[j] = delta_output * weights_hidden5_output[j] * (1 - hidden5_output[j])
                end

                delta_hidden4 = {}
                for j = 1, hidden4_size do
                    delta_hidden4[j] = 0
                    for k = 1, hidden5_size do
                        delta_hidden4[j] = delta_hidden4[j]
                            + delta_hidden5[k] * weights_hidden4_hidden5[j][k] * (1 - hidden4_input[j])
                    end
                end

                delta_hidden3 = {}
                for j = 1, hidden3_size do
                    delta_hidden3[j] = 0
                    for k = 1, hidden4_size do
                        delta_hidden3[j] = delta_hidden3[j]
                            + delta_hidden4[k] * weights_hidden3_hidden4[j][k] * (1 - hidden3_input[j])
                    end
                end

                delta_hidden2 = {}
                for j = 1, hidden2_size do
                    delta_hidden2[j] = 0
                    for k = 1, hidden3_size do
                        delta_hidden2[j] = delta_hidden2[j]
                            + delta_hidden3[k] * weights_hidden2_hidden3[j][k] * (1 - hidden2_input[j])
                    end
                end

                delta_hidden1 = {}
                for j = 1, hidden1_size do
                    delta_hidden1[j] = 0
                    for k = 1, hidden2_size do
                        delta_hidden1[j] = delta_hidden1[j]
                            + delta_hidden2[k] * weights_hidden1_hidden2[j][k] * (1 - hidden1_input[j])
                    end
                end

                -- Aktualizacja wag
                for j = 1, hidden5_size do
                    weights_hidden5_output[j] = weights_hidden5_output[j]
                        + learning_rate * delta_output * hidden5_output[j]
                end

                for j = 1, hidden4_size do
                    for k = 1, hidden5_size do
                        weights_hidden4_hidden5[j][k] = weights_hidden4_hidden5[j][k]
                            + learning_rate * delta_hidden5[k] * hidden4_input[j]
                    end
                end

                for j = 1, hidden3_size do
                    for k = 1, hidden4_size do
                        weights_hidden3_hidden4[j][k] = weights_hidden3_hidden4[j][k]
                            + learning_rate * delta_hidden4[k] * hidden3_input[j]
                    end
                end

                for j = 1, hidden2_size do
                    for k = 1, hidden3_size do
                        weights_hidden2_hidden3[j][k] = weights_hidden2_hidden3[j][k]
                            + learning_rate * delta_hidden3[k] * hidden2_input[j]
                    end
                end

                for j = 1, hidden1_size do
                    for k = 1, hidden2_size do
                        weights_hidden1_hidden2[j][k] = weights_hidden1_hidden2[j][k]
                            + learning_rate * delta_hidden2[k] * hidden1_input[j]
                    end
                end

                for j = 1, input_size do
                    for k = 1, hidden1_size do
                        weights_input_hidden1[j][k] = weights_input_hidden1[j][k]
                            + learning_rate * delta_hidden1[k] * x_train[i][j]
                    end
                end
            end
        end

        -- Testowanie sieci

        for i = 1, #x_test do
            print("")

            hidden1_input = {}
            hidden2_input = {}
            hidden3_input = {}
            hidden4_input = {}
            hidden5_input = {}
            hidden5_output = {}
            final_input = {}

            for j = 1, hidden1_size do
                hidden1_input[j] = 0
                for k = 1, input_size do
                    hidden1_input[j] = hidden1_input[j] + x_test[i][k] * weights_input_hidden1[k][j]
                end
                hidden1_input[j] = activation_input(hidden1_input[j])
            end

            for j = 1, hidden2_size do
                hidden2_input[j] = 0
                for k = 1, hidden1_size do
                    hidden2_input[j] = hidden2_input[j] + hidden1_input[k] * weights_hidden1_hidden2[k][j]
                end
                hidden2_input[j] = activation_hd(hidden2_input[j])
            end

            for j = 1, hidden3_size do
                hidden3_input[j] = 0
                for k = 1, hidden2_size do
                    hidden3_input[j] = hidden3_input[j] + hidden2_input[k] * weights_hidden2_hidden3[k][j]
                end
                hidden3_input[j] = activation_hd(hidden3_input[j])
            end

            for j = 1, hidden4_size do
                hidden4_input[j] = 0
                for k = 1, hidden3_size do
                    hidden4_input[j] = hidden4_input[j] + hidden3_input[k] * weights_hidden3_hidden4[k][j]
                end
                hidden4_input[j] = activation_hd(hidden4_input[j])
            end

            for j = 1, hidden5_size do
                hidden5_input[j] = 0
                for k = 1, hidden4_size do
                    hidden5_input[j] = hidden5_input[j] + hidden4_input[k] * weights_hidden4_hidden5[k][j]
                end
                hidden5_input[j] = activation_hd(hidden5_input[j])
            end

            for j = 1, hidden5_size do
                hidden5_output[j] = activation_hd(hidden5_input[j])
            end

            final_input[1] = 0
            for j = 1, hidden5_size do
                final_input[1] = final_input[1] + hidden5_output[j] * weights_hidden5_output[j]
            end
            final_output = activation_output(final_input[1])

            print("Test input:      " .. grey, table.concat(x_test[i], " ") .. reset)
            print("Predicted output:" .. red, final_output .. reset)
        end
    end

    if hidden_layers == 6 then
        weights_input_hidden1 = {}
        weights_hidden1_hidden2 = {}
        weights_hidden2_hidden3 = {}
        weights_hidden3_hidden4 = {}
        weights_hidden4_hidden5 = {}
        weights_hidden5_hidden6 = {}
        weights_hidden6_output = {}

        for i = 1, input_size do
            weights_input_hidden1[i] = {}
            for j = 1, hidden1_size do
                weights_input_hidden1[i][j] = math.random()
            end
        end

        for i = 1, hidden1_size do
            weights_hidden1_hidden2[i] = {}
            for j = 1, hidden2_size do
                weights_hidden1_hidden2[i][j] = math.random()
            end
        end

        for i = 1, hidden2_size do
            weights_hidden2_hidden3[i] = {}
            for j = 1, hidden3_size do
                weights_hidden2_hidden3[i][j] = math.random()
            end
        end

        for i = 1, hidden3_size do
            weights_hidden3_hidden4[i] = {}
            for j = 1, hidden4_size do
                weights_hidden3_hidden4[i][j] = math.random()
            end
        end

        for i = 1, hidden4_size do
            weights_hidden4_hidden5[i] = {}
            for j = 1, hidden5_size do
                weights_hidden4_hidden5[i][j] = math.random()
            end
        end

        for i = 1, hidden5_size do
            weights_hidden5_hidden6[i] = {}
            for j = 1, hidden6_size do
                weights_hidden5_hidden6[i][j] = math.random()
            end
        end

        for i = 1, hidden6_size do
            weights_hidden6_output[i] = math.random()
        end

        -- Uczenie sieci

        epochs = iterations

        for epoch = 1, epochs do
            for i = 1, #x_train do
                -- Forward pass
                hidden1_input = {}
                hidden2_input = {}
                hidden3_input = {}
                hidden4_input = {}
                hidden5_input = {}
                hidden6_input = {}
                hidden6_output = {}
                final_input = {}

                for j = 1, hidden1_size do
                    hidden1_input[j] = 0
                    for k = 1, input_size do
                        hidden1_input[j] = hidden1_input[j] + x_train[i][k] * weights_input_hidden1[k][j]
                    end
                    hidden1_input[j] = activation_input(hidden1_input[j])
                end

                for j = 1, hidden2_size do
                    hidden2_input[j] = 0
                    for k = 1, hidden1_size do
                        hidden2_input[j] = hidden2_input[j] + hidden1_input[k] * weights_hidden1_hidden2[k][j]
                    end
                    hidden2_input[j] = activation_hd(hidden2_input[j])
                end

                for j = 1, hidden3_size do
                    hidden3_input[j] = 0
                    for k = 1, hidden2_size do
                        hidden3_input[j] = hidden3_input[j] + hidden2_input[k] * weights_hidden2_hidden3[k][j]
                    end
                    hidden3_input[j] = activation_hd(hidden3_input[j])
                end

                for j = 1, hidden4_size do
                    hidden4_input[j] = 0
                    for k = 1, hidden3_size do
                        hidden4_input[j] = hidden4_input[j] + hidden3_input[k] * weights_hidden3_hidden4[k][j]
                    end
                    hidden4_input[j] = activation_hd(hidden4_input[j])
                end

                for j = 1, hidden5_size do
                    hidden5_input[j] = 0
                    for k = 1, hidden4_size do
                        hidden5_input[j] = hidden5_input[j] + hidden4_input[k] * weights_hidden4_hidden5[k][j]
                    end
                    hidden5_input[j] = activation_hd(hidden5_input[j])
                end

                for j = 1, hidden6_size do
                    hidden6_input[j] = 0
                    for k = 1, hidden5_size do
                        hidden6_input[j] = hidden6_input[j] + hidden5_input[k] * weights_hidden5_hidden6[k][j]
                    end
                    hidden6_input[j] = activation_hd(hidden6_input[j])
                end

                for j = 1, hidden6_size do
                    hidden6_output[j] = activation_hd(hidden6_input[j])
                end

                final_input[1] = 0
                for j = 1, hidden6_size do
                    final_input[1] = final_input[1] + hidden6_output[j] * weights_hidden6_output[j]
                end
                final_output = activation_output(final_input[1])

                -- Obliczenie błędu
                error = y_train[i] - final_output

                -- Backpropagation
                delta_output = error * final_output * (1 - final_output)

                delta_hidden6 = {}
                for j = 1, hidden6_size do
                    delta_hidden6[j] = delta_output * weights_hidden6_output[j] * (1 - hidden6_output[j])
                end

                delta_hidden5 = {}
                for j = 1, hidden5_size do
                    delta_hidden5[j] = 0
                    for k = 1, hidden6_size do
                        delta_hidden5[j] = delta_hidden5[j]
                            + delta_hidden6[k] * weights_hidden5_hidden6[j][k] * (1 - hidden5_input[j])
                    end
                end

                delta_hidden4 = {}
                for j = 1, hidden4_size do
                    delta_hidden4[j] = 0
                    for k = 1, hidden5_size do
                        delta_hidden4[j] = delta_hidden4[j]
                            + delta_hidden5[k] * weights_hidden4_hidden5[j][k] * (1 - hidden4_input[j])
                    end
                end

                delta_hidden3 = {}
                for j = 1, hidden3_size do
                    delta_hidden3[j] = 0
                    for k = 1, hidden4_size do
                        delta_hidden3[j] = delta_hidden3[j]
                            + delta_hidden4[k] * weights_hidden3_hidden4[j][k] * (1 - hidden3_input[j])
                    end
                end

                delta_hidden2 = {}
                for j = 1, hidden2_size do
                    delta_hidden2[j] = 0
                    for k = 1, hidden3_size do
                        delta_hidden2[j] = delta_hidden2[j]
                            + delta_hidden3[k] * weights_hidden2_hidden3[j][k] * (1 - hidden2_input[j])
                    end
                end

                delta_hidden1 = {}
                for j = 1, hidden1_size do
                    delta_hidden1[j] = 0
                    for k = 1, hidden2_size do
                        delta_hidden1[j] = delta_hidden1[j]
                            + delta_hidden2[k] * weights_hidden1_hidden2[j][k] * (1 - hidden1_input[j])
                    end
                end

                -- Aktualizacja wag
                for j = 1, hidden6_size do
                    weights_hidden6_output[j] = weights_hidden6_output[j]
                        + learning_rate * delta_output * hidden6_output[j]
                end

                for j = 1, hidden5_size do
                    for k = 1, hidden6_size do
                        weights_hidden5_hidden6[j][k] = weights_hidden5_hidden6[j][k]
                            + learning_rate * delta_hidden6[k] * hidden5_input[j]
                    end
                end

                for j = 1, hidden4_size do
                    for k = 1, hidden5_size do
                        weights_hidden4_hidden5[j][k] = weights_hidden4_hidden5[j][k]
                            + learning_rate * delta_hidden5[k] * hidden4_input[j]
                    end
                end

                for j = 1, hidden3_size do
                    for k = 1, hidden4_size do
                        weights_hidden3_hidden4[j][k] = weights_hidden3_hidden4[j][k]
                            + learning_rate * delta_hidden4[k] * hidden3_input[j]
                    end
                end

                for j = 1, hidden2_size do
                    for k = 1, hidden3_size do
                        weights_hidden2_hidden3[j][k] = weights_hidden2_hidden3[j][k]
                            + learning_rate * delta_hidden3[k] * hidden2_input[j]
                    end
                end

                for j = 1, hidden1_size do
                    for k = 1, hidden2_size do
                        weights_hidden1_hidden2[j][k] = weights_hidden1_hidden2[j][k]
                            + learning_rate * delta_hidden2[k] * hidden1_input[j]
                    end
                end

                for j = 1, input_size do
                    for k = 1, hidden1_size do
                        weights_input_hidden1[j][k] = weights_input_hidden1[j][k]
                            + learning_rate * delta_hidden1[k] * x_train[i][j]
                    end
                end
            end
        end

        -- Testowanie sieci

        for i = 1, #x_test do
            print("")

            hidden1_input = {}
            hidden2_input = {}
            hidden3_input = {}
            hidden4_input = {}
            hidden5_input = {}
            hidden6_input = {}
            hidden6_output = {}
            final_input = {}

            for j = 1, hidden1_size do
                hidden1_input[j] = 0
                for k = 1, input_size do
                    hidden1_input[j] = hidden1_input[j] + x_test[i][k] * weights_input_hidden1[k][j]
                end
                hidden1_input[j] = activation_input(hidden1_input[j])
            end

            for j = 1, hidden2_size do
                hidden2_input[j] = 0
                for k = 1, hidden1_size do
                    hidden2_input[j] = hidden2_input[j] + hidden1_input[k] * weights_hidden1_hidden2[k][j]
                end
                hidden2_input[j] = activation_hd(hidden2_input[j])
            end

            for j = 1, hidden3_size do
                hidden3_input[j] = 0
                for k = 1, hidden2_size do
                    hidden3_input[j] = hidden3_input[j] + hidden2_input[k] * weights_hidden2_hidden3[k][j]
                end
                hidden3_input[j] = activation_hd(hidden3_input[j])
            end

            for j = 1, hidden4_size do
                hidden4_input[j] = 0
                for k = 1, hidden3_size do
                    hidden4_input[j] = hidden4_input[j] + hidden3_input[k] * weights_hidden3_hidden4[k][j]
                end
                hidden4_input[j] = activation_hd(hidden4_input[j])
            end

            for j = 1, hidden5_size do
                hidden5_input[j] = 0
                for k = 1, hidden4_size do
                    hidden5_input[j] = hidden5_input[j] + hidden4_input[k] * weights_hidden4_hidden5[k][j]
                end
                hidden5_input[j] = activation_hd(hidden5_input[j])
            end

            for j = 1, hidden6_size do
                hidden6_input[j] = 0
                for k = 1, hidden5_size do
                    hidden6_input[j] = hidden6_input[j] + hidden5_input[k] * weights_hidden5_hidden6[k][j]
                end
                hidden6_input[j] = activation_hd(hidden6_input[j])
            end

            for j = 1, hidden6_size do
                hidden6_output[j] = activation_hd(hidden6_input[j])
            end

            final_input[1] = 0
            for j = 1, hidden6_size do
                final_input[1] = final_input[1] + hidden6_output[j] * weights_hidden6_output[j]
            end
            final_output = activation_output(final_input[1])

            print("Test input:      " .. grey, table.concat(x_test[i], " ") .. reset)
            print("Predicted output:" .. red, final_output .. reset)
        end
    end

    if hidden_layers == 7 then
        weights_input_hidden1 = {}
        weights_hidden1_hidden2 = {}
        weights_hidden2_hidden3 = {}
        weights_hidden3_hidden4 = {}
        weights_hidden4_hidden5 = {}
        weights_hidden5_hidden6 = {}
        weights_hidden6_hidden7 = {}
        weights_hidden7_output = {}

        for i = 1, input_size do
            weights_input_hidden1[i] = {}
            for j = 1, hidden1_size do
                weights_input_hidden1[i][j] = math.random()
            end
        end

        for i = 1, hidden1_size do
            weights_hidden1_hidden2[i] = {}
            for j = 1, hidden2_size do
                weights_hidden1_hidden2[i][j] = math.random()
            end
        end

        for i = 1, hidden2_size do
            weights_hidden2_hidden3[i] = {}
            for j = 1, hidden3_size do
                weights_hidden2_hidden3[i][j] = math.random()
            end
        end

        for i = 1, hidden3_size do
            weights_hidden3_hidden4[i] = {}
            for j = 1, hidden4_size do
                weights_hidden3_hidden4[i][j] = math.random()
            end
        end

        for i = 1, hidden4_size do
            weights_hidden4_hidden5[i] = {}
            for j = 1, hidden5_size do
                weights_hidden4_hidden5[i][j] = math.random()
            end
        end

        for i = 1, hidden5_size do
            weights_hidden5_hidden6[i] = {}
            for j = 1, hidden6_size do
                weights_hidden5_hidden6[i][j] = math.random()
            end
        end

        for i = 1, hidden6_size do
            weights_hidden6_hidden7[i] = {}
            for j = 1, hidden7_size do
                weights_hidden6_hidden7[i][j] = math.random()
            end
        end

        for i = 1, hidden7_size do
            weights_hidden7_output[i] = math.random()
        end

        -- Uczenie sieci

        epochs = iterations

        for epoch = 1, epochs do
            for i = 1, #x_train do
                -- Forward pass
                hidden1_input = {}
                hidden2_input = {}
                hidden3_input = {}
                hidden4_input = {}
                hidden5_input = {}
                hidden6_input = {}
                hidden7_input = {}
                hidden7_output = {}
                final_input = {}

                for j = 1, hidden1_size do
                    hidden1_input[j] = 0
                    for k = 1, input_size do
                        hidden1_input[j] = hidden1_input[j] + x_train[i][k] * weights_input_hidden1[k][j]
                    end
                    hidden1_input[j] = activation_input(hidden1_input[j])
                end

                for j = 1, hidden2_size do
                    hidden2_input[j] = 0
                    for k = 1, hidden1_size do
                        hidden2_input[j] = hidden2_input[j] + hidden1_input[k] * weights_hidden1_hidden2[k][j]
                    end
                    hidden2_input[j] = activation_hd(hidden2_input[j])
                end

                for j = 1, hidden3_size do
                    hidden3_input[j] = 0
                    for k = 1, hidden2_size do
                        hidden3_input[j] = hidden3_input[j] + hidden2_input[k] * weights_hidden2_hidden3[k][j]
                    end
                    hidden3_input[j] = activation_hd(hidden3_input[j])
                end

                for j = 1, hidden4_size do
                    hidden4_input[j] = 0
                    for k = 1, hidden3_size do
                        hidden4_input[j] = hidden4_input[j] + hidden3_input[k] * weights_hidden3_hidden4[k][j]
                    end
                    hidden4_input[j] = activation_hd(hidden4_input[j])
                end

                for j = 1, hidden5_size do
                    hidden5_input[j] = 0
                    for k = 1, hidden4_size do
                        hidden5_input[j] = hidden5_input[j] + hidden4_input[k] * weights_hidden4_hidden5[k][j]
                    end
                    hidden5_input[j] = activation_hd(hidden5_input[j])
                end

                for j = 1, hidden6_size do
                    hidden6_input[j] = 0
                    for k = 1, hidden5_size do
                        hidden6_input[j] = hidden6_input[j] + hidden5_input[k] * weights_hidden5_hidden6[k][j]
                    end
                    hidden6_input[j] = activation_hd(hidden6_input[j])
                end

                for j = 1, hidden7_size do
                    hidden7_input[j] = 0
                    for k = 1, hidden6_size do
                        hidden7_input[j] = hidden7_input[j] + hidden6_input[k] * weights_hidden6_hidden7[k][j]
                    end
                    hidden7_input[j] = activation_hd(hidden7_input[j])
                end

                for j = 1, hidden7_size do
                    hidden7_output[j] = activation_hd(hidden7_input[j])
                end

                final_input[1] = 0
                for j = 1, hidden7_size do
                    final_input[1] = final_input[1] + hidden7_output[j] * weights_hidden7_output[j]
                end
                final_output = activation_output(final_input[1])

                -- Obliczenie błędu
                error = y_train[i] - final_output

                -- Backpropagation
                delta_output = error * final_output * (1 - final_output)

                delta_hidden7 = {}
                for j = 1, hidden7_size do
                    delta_hidden7[j] = delta_output * weights_hidden7_output[j] * (1 - hidden7_output[j])
                end

                delta_hidden6 = {}
                for j = 1, hidden6_size do
                    delta_hidden6[j] = 0
                    for k = 1, hidden7_size do
                        delta_hidden6[j] = delta_hidden6[j]
                            + delta_hidden7[k] * weights_hidden6_hidden7[j][k] * (1 - hidden6_input[j])
                    end
                end

                delta_hidden5 = {}
                for j = 1, hidden5_size do
                    delta_hidden5[j] = 0
                    for k = 1, hidden6_size do
                        delta_hidden5[j] = delta_hidden5[j]
                            + delta_hidden6[k] * weights_hidden5_hidden6[j][k] * (1 - hidden5_input[j])
                    end
                end

                delta_hidden4 = {}
                for j = 1, hidden4_size do
                    delta_hidden4[j] = 0
                    for k = 1, hidden5_size do
                        delta_hidden4[j] = delta_hidden4[j]
                            + delta_hidden5[k] * weights_hidden4_hidden5[j][k] * (1 - hidden4_input[j])
                    end
                end

                delta_hidden3 = {}
                for j = 1, hidden3_size do
                    delta_hidden3[j] = 0
                    for k = 1, hidden4_size do
                        delta_hidden3[j] = delta_hidden3[j]
                            + delta_hidden4[k] * weights_hidden3_hidden4[j][k] * (1 - hidden3_input[j])
                    end
                end

                delta_hidden2 = {}
                for j = 1, hidden2_size do
                    delta_hidden2[j] = 0
                    for k = 1, hidden3_size do
                        delta_hidden2[j] = delta_hidden2[j]
                            + delta_hidden3[k] * weights_hidden2_hidden3[j][k] * (1 - hidden2_input[j])
                    end
                end

                delta_hidden1 = {}
                for j = 1, hidden1_size do
                    delta_hidden1[j] = 0
                    for k = 1, hidden2_size do
                        delta_hidden1[j] = delta_hidden1[j]
                            + delta_hidden2[k] * weights_hidden1_hidden2[j][k] * (1 - hidden1_input[j])
                    end
                end

                -- Aktualizacja wag
                for j = 1, hidden7_size do
                    weights_hidden7_output[j] = weights_hidden7_output[j]
                        + learning_rate * delta_output * hidden7_output[j]
                end

                for j = 1, hidden6_size do
                    for k = 1, hidden7_size do
                        weights_hidden6_hidden7[j][k] = weights_hidden6_hidden7[j][k]
                            + learning_rate * delta_hidden7[k] * hidden6_input[j]
                    end
                end

                for j = 1, hidden5_size do
                    for k = 1, hidden6_size do
                        weights_hidden5_hidden6[j][k] = weights_hidden5_hidden6[j][k]
                            + learning_rate * delta_hidden6[k] * hidden5_input[j]
                    end
                end

                for j = 1, hidden4_size do
                    for k = 1, hidden5_size do
                        weights_hidden4_hidden5[j][k] = weights_hidden4_hidden5[j][k]
                            + learning_rate * delta_hidden5[k] * hidden4_input[j]
                    end
                end

                for j = 1, hidden3_size do
                    for k = 1, hidden4_size do
                        weights_hidden3_hidden4[j][k] = weights_hidden3_hidden4[j][k]
                            + learning_rate * delta_hidden4[k] * hidden3_input[j]
                    end
                end

                for j = 1, hidden2_size do
                    for k = 1, hidden3_size do
                        weights_hidden2_hidden3[j][k] = weights_hidden2_hidden3[j][k]
                            + learning_rate * delta_hidden3[k] * hidden2_input[j]
                    end
                end

                for j = 1, hidden1_size do
                    for k = 1, hidden2_size do
                        weights_hidden1_hidden2[j][k] = weights_hidden1_hidden2[j][k]
                            + learning_rate * delta_hidden2[k] * hidden1_input[j]
                    end
                end

                for j = 1, input_size do
                    for k = 1, hidden1_size do
                        weights_input_hidden1[j][k] = weights_input_hidden1[j][k]
                            + learning_rate * delta_hidden1[k] * x_train[i][j]
                    end
                end
            end
        end

        -- Testowanie sieci

        for i = 1, #x_test do
            print("")

            hidden1_input = {}
            hidden2_input = {}
            hidden3_input = {}
            hidden4_input = {}
            hidden5_input = {}
            hidden6_input = {}
            hidden7_input = {}
            hidden7_output = {}
            final_input = {}

            for j = 1, hidden1_size do
                hidden1_input[j] = 0
                for k = 1, input_size do
                    hidden1_input[j] = hidden1_input[j] + x_test[i][k] * weights_input_hidden1[k][j]
                end
                hidden1_input[j] = activation_input(hidden1_input[j])
            end

            for j = 1, hidden2_size do
                hidden2_input[j] = 0
                for k = 1, hidden1_size do
                    hidden2_input[j] = hidden2_input[j] + hidden1_input[k] * weights_hidden1_hidden2[k][j]
                end
                hidden2_input[j] = activation_hd(hidden2_input[j])
            end

            for j = 1, hidden3_size do
                hidden3_input[j] = 0
                for k = 1, hidden2_size do
                    hidden3_input[j] = hidden3_input[j] + hidden2_input[k] * weights_hidden2_hidden3[k][j]
                end
                hidden3_input[j] = activation_hd(hidden3_input[j])
            end

            for j = 1, hidden4_size do
                hidden4_input[j] = 0
                for k = 1, hidden3_size do
                    hidden4_input[j] = hidden4_input[j] + hidden3_input[k] * weights_hidden3_hidden4[k][j]
                end
                hidden4_input[j] = activation_hd(hidden4_input[j])
            end

            for j = 1, hidden5_size do
                hidden5_input[j] = 0
                for k = 1, hidden4_size do
                    hidden5_input[j] = hidden5_input[j] + hidden4_input[k] * weights_hidden4_hidden5[k][j]
                end
                hidden5_input[j] = activation_hd(hidden5_input[j])
            end

            for j = 1, hidden6_size do
                hidden6_input[j] = 0
                for k = 1, hidden5_size do
                    hidden6_input[j] = hidden6_input[j] + hidden5_input[k] * weights_hidden5_hidden6[k][j]
                end
                hidden6_input[j] = activation_hd(hidden6_input[j])
            end

            for j = 1, hidden7_size do
                hidden7_input[j] = 0
                for k = 1, hidden6_size do
                    hidden7_input[j] = hidden7_input[j] + hidden6_input[k] * weights_hidden6_hidden7[k][j]
                end
                hidden7_input[j] = activation_hd(hidden7_input[j])
            end

            for j = 1, hidden7_size do
                hidden7_output[j] = activation_hd(hidden7_input[j])
            end

            final_input[1] = 0
            for j = 1, hidden7_size do
                final_input[1] = final_input[1] + hidden7_output[j] * weights_hidden7_output[j]
            end
            final_output = activation_output(final_input[1])

            print("Test input:      " .. grey, table.concat(x_test[i], " ") .. reset)
            print("Predicted output:" .. red, final_output .. reset)
        end
    end
end

if optimisation == "adamw" or optimisation == "AdamW" or optimisation == "ADAMW" then
    if hidden_layers == 2 then
        weights_input_hidden1 = {}
        weights_hidden1_hidden2 = {}
        weights_hidden2_output = {}

        for i = 1, input_size do
            weights_input_hidden1[i] = {}
            for j = 1, hidden1_size do
                weights_input_hidden1[i][j] = math.random()
            end
        end

        for i = 1, hidden1_size do
            weights_hidden1_hidden2[i] = {}
            for j = 1, hidden2_size do
                weights_hidden1_hidden2[i][j] = math.random()
            end
        end

        for i = 1, hidden2_size do
            weights_hidden2_output[i] = math.random()
        end

        -- Inicjalizacja zmiennych AdamW
        m_input_hidden1 = {}
        v_input_hidden1 = {}
        m_hidden1_hidden2 = {}
        v_hidden1_hidden2 = {}
        m_hidden2_output = {}
        v_hidden2_output = {}
        wd_input_hidden1 = {}
        wd_hidden1_hidden2 = {}
        wd_hidden2_output = {}

        for i = 1, input_size do
            m_input_hidden1[i] = {}
            v_input_hidden1[i] = {}
            wd_input_hidden1[i] = {}
            for j = 1, hidden1_size do
                m_input_hidden1[i][j] = 0
                v_input_hidden1[i][j] = 0
                wd_input_hidden1[i][j] = 0
            end
        end

        for i = 1, hidden1_size do
            m_hidden1_hidden2[i] = {}
            v_hidden1_hidden2[i] = {}
            wd_hidden1_hidden2[i] = {}
            for j = 1, hidden2_size do
                m_hidden1_hidden2[i][j] = 0
                v_hidden1_hidden2[i][j] = 0
                wd_hidden1_hidden2[i][j] = 0
            end
        end

        for i = 1, hidden2_size do
            m_hidden2_output[i] = 0
            v_hidden2_output[i] = 0
            wd_hidden2_output[i] = 0
        end

        -- Uczenie sieci
        epochs = iterations
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        learning_rate = 0.001

        for epoch = 1, epochs do
            for i = 1, #x_train do
                -- Forward pass
                hidden1_input = {}
                hidden2_input = {}
                hidden2_output = {}
                final_input = {}

                for j = 1, hidden1_size do
                    hidden1_input[j] = 0
                    for k = 1, input_size do
                        hidden1_input[j] = hidden1_input[j] + x_train[i][k] * weights_input_hidden1[k][j]
                    end
                    hidden1_input[j] = activation_input(hidden1_input[j])
                end

                for j = 1, hidden2_size do
                    hidden2_input[j] = 0
                    for k = 1, hidden1_size do
                        hidden2_input[j] = hidden2_input[j] + hidden1_input[k] * weights_hidden1_hidden2[k][j]
                    end
                    hidden2_input[j] = activation_hd(hidden2_input[j])
                end

                for j = 1, hidden2_size do
                    hidden2_output[j] = activation_hd(hidden2_input[j])
                end

                final_input[1] = 0
                for j = 1, hidden2_size do
                    final_input[1] = final_input[1] + hidden2_output[j] * weights_hidden2_output[j]
                end
                final_output = activation_output(final_input[1])

                -- Obliczenie błędu
                error = y_train[i] - final_output

                -- Backpropagation
                delta_output = error * final_output * (1 - final_output)

                delta_hidden2 = {}
                for j = 1, hidden2_size do
                    delta_hidden2[j] = delta_output * weights_hidden2_output[j] * (1 - hidden2_output[j])
                end

                delta_hidden1 = {}
                for j = 1, hidden1_size do
                    delta_hidden1[j] = 0
                    for k = 1, hidden2_size do
                        delta_hidden1[j] = delta_hidden1[j]
                            + delta_hidden2[k] * weights_hidden1_hidden2[j][k] * (1 - hidden1_input[j])
                    end
                end

                -- Aktualizacja wag zgodnie z AdamW
                for j = 1, hidden2_size do
                    m_hidden2_output[j] = beta1 * m_hidden2_output[j] + (1 - beta1) * delta_output * hidden2_output[j]
                    v_hidden2_output[j] = beta2 * v_hidden2_output[j]
                        + (1 - beta2) * (delta_output * hidden2_output[j]) ^ 2
                    m_hat = m_hidden2_output[j] / (1 - beta1 ^ epoch)
                    v_hat = v_hidden2_output[j] / (1 - beta2 ^ epoch)
                    wd_hidden2_output[j] = weight_decay * weights_hidden2_output[j]
                    weights_hidden2_output[j] = weights_hidden2_output[j]
                        + learning_rate * (m_hat / (math.sqrt(v_hat) + epsilon) - wd_hidden2_output[j])
                end

                for j = 1, hidden1_size do
                    for k = 1, hidden2_size do
                        m_hidden1_hidden2[j][k] = beta1 * m_hidden1_hidden2[j][k]
                            + (1 - beta1) * delta_hidden2[k] * hidden1_input[j]
                        v_hidden1_hidden2[j][k] = beta2 * v_hidden1_hidden2[j][k]
                            + (1 - beta2) * (delta_hidden2[k] * hidden1_input[j]) ^ 2
                        m_hat = m_hidden1_hidden2[j][k] / (1 - beta1 ^ epoch)
                        v_hat = v_hidden1_hidden2[j][k] / (1 - beta2 ^ epoch)
                        wd_hidden1_hidden2[j][k] = weight_decay * weights_hidden1_hidden2[j][k]
                        weights_hidden1_hidden2[j][k] = weights_hidden1_hidden2[j][k]
                            + learning_rate * (m_hat / (math.sqrt(v_hat) + epsilon) - wd_hidden1_hidden2[j][k])
                    end
                end

                for j = 1, input_size do
                    for k = 1, hidden1_size do
                        m_input_hidden1[j][k] = beta1 * m_input_hidden1[j][k]
                            + (1 - beta1) * delta_hidden1[k] * x_train[i][j]
                        v_input_hidden1[j][k] = beta2 * v_input_hidden1[j][k]
                            + (1 - beta2) * (delta_hidden1[k] * x_train[i][j]) ^ 2
                        m_hat = m_input_hidden1[j][k] / (1 - beta1 ^ epoch)
                        v_hat = v_input_hidden1[j][k] / (1 - beta2 ^ epoch)
                        wd_input_hidden1[j][k] = weight_decay * weights_input_hidden1[j][k]
                        weights_input_hidden1[j][k] = weights_input_hidden1[j][k]
                            + learning_rate * (m_hat / (math.sqrt(v_hat) + epsilon) - wd_input_hidden1[j][k])
                    end
                end
            end
        end

        -- Testowanie sieci
        for i = 1, #x_test do
            print("")

            hidden1_input = {}
            hidden2_input = {}
            hidden2_output = {}
            final_input = {}

            for j = 1, hidden1_size do
                hidden1_input[j] = 0
                for k = 1, input_size do
                    hidden1_input[j] = hidden1_input[j] + x_test[i][k] * weights_input_hidden1[k][j]
                end
                hidden1_input[j] = activation_input(hidden1_input[j])
            end

            for j = 1, hidden2_size do
                hidden2_input[j] = 0
                for k = 1, hidden1_size do
                    hidden2_input[j] = hidden2_input[j] + hidden1_input[k] * weights_hidden1_hidden2[k][j]
                end
                hidden2_input[j] = activation_hd(hidden2_input[j])
            end

            for j = 1, hidden2_size do
                hidden2_output[j] = activation_hd(hidden2_input[j])
            end

            final_input[1] = 0
            for j = 1, hidden2_size do
                final_input[1] = final_input[1] + hidden2_output[j] * weights_hidden2_output[j]
            end
            final_output = activation_output(final_input[1])

            print("Test input:      " .. grey, table.concat(x_test[i], " ") .. reset)
            print("Predicted output:" .. red, final_output .. reset)
        end
    end

    if hidden_layers == 3 then
        weights_input_hidden1 = {}
        weights_hidden1_hidden2 = {}
        weights_hidden2_hidden3 = {}
        weights_hidden3_output = {}

        for i = 1, input_size do
            weights_input_hidden1[i] = {}
            for j = 1, hidden1_size do
                weights_input_hidden1[i][j] = math.random()
            end
        end

        for i = 1, hidden1_size do
            weights_hidden1_hidden2[i] = {}
            for j = 1, hidden2_size do
                weights_hidden1_hidden2[i][j] = math.random()
            end
        end

        for i = 1, hidden2_size do
            weights_hidden2_hidden3[i] = {}
            for j = 1, hidden3_size do
                weights_hidden2_hidden3[i][j] = math.random()
            end
        end

        for i = 1, hidden3_size do
            weights_hidden3_output[i] = math.random()
        end

        -- Inicjalizacja zmiennych AdamW
        m_input_hidden1 = {}
        v_input_hidden1 = {}
        m_hidden1_hidden2 = {}
        v_hidden1_hidden2 = {}
        m_hidden2_hidden3 = {}
        v_hidden2_hidden3 = {}
        m_hidden3_output = {}
        v_hidden3_output = {}
        wd_input_hidden1 = {}
        wd_hidden1_hidden2 = {}
        wd_hidden2_hidden3 = {}
        wd_hidden3_output = {}

        for i = 1, input_size do
            m_input_hidden1[i] = {}
            v_input_hidden1[i] = {}
            wd_input_hidden1[i] = {}
            for j = 1, hidden1_size do
                m_input_hidden1[i][j] = 0
                v_input_hidden1[i][j] = 0
                wd_input_hidden1[i][j] = 0
            end
        end

        for i = 1, hidden1_size do
            m_hidden1_hidden2[i] = {}
            v_hidden1_hidden2[i] = {}
            wd_hidden1_hidden2[i] = {}
            for j = 1, hidden2_size do
                m_hidden1_hidden2[i][j] = 0
                v_hidden1_hidden2[i][j] = 0
                wd_hidden1_hidden2[i][j] = 0
            end
        end

        for i = 1, hidden2_size do
            m_hidden2_hidden3[i] = {}
            v_hidden2_hidden3[i] = {}
            wd_hidden2_hidden3[i] = {}
            for j = 1, hidden3_size do
                m_hidden2_hidden3[i][j] = 0
                v_hidden2_hidden3[i][j] = 0
                wd_hidden2_hidden3[i][j] = 0
            end
        end

        for i = 1, hidden3_size do
            m_hidden3_output[i] = 0
            v_hidden3_output[i] = 0
            wd_hidden3_output[i] = 0
        end

        -- Uczenie sieci
        epochs = iterations
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        learning_rate = 0.001

        for epoch = 1, epochs do
            for i = 1, #x_train do
                -- Forward pass
                hidden1_input = {}
                hidden2_input = {}
                hidden3_input = {}
                hidden3_output = {}
                final_input = {}

                for j = 1, hidden1_size do
                    hidden1_input[j] = 0
                    for k = 1, input_size do
                        hidden1_input[j] = hidden1_input[j] + x_train[i][k] * weights_input_hidden1[k][j]
                    end
                    hidden1_input[j] = activation_input(hidden1_input[j])
                end

                for j = 1, hidden2_size do
                    hidden2_input[j] = 0
                    for k = 1, hidden1_size do
                        hidden2_input[j] = hidden2_input[j] + hidden1_input[k] * weights_hidden1_hidden2[k][j]
                    end
                    hidden2_input[j] = activation_hd(hidden2_input[j])
                end

                for j = 1, hidden3_size do
                    hidden3_input[j] = 0
                    for k = 1, hidden2_size do
                        hidden3_input[j] = hidden3_input[j] + hidden2_input[k] * weights_hidden2_hidden3[k][j]
                    end
                    hidden3_input[j] = activation_hd(hidden3_input[j])
                end

                for j = 1, hidden3_size do
                    hidden3_output[j] = activation_hd(hidden3_input[j])
                end

                final_input[1] = 0
                for j = 1, hidden3_size do
                    final_input[1] = final_input[1] + hidden3_output[j] * weights_hidden3_output[j]
                end
                final_output = activation_output(final_input[1])

                -- Obliczenie błędu
                error = y_train[i] - final_output

                -- Backpropagation
                delta_output = error * final_output * (1 - final_output)

                delta_hidden3 = {}
                for j = 1, hidden3_size do
                    delta_hidden3[j] = delta_output * weights_hidden3_output[j] * (1 - hidden3_output[j])
                end

                delta_hidden2 = {}
                for j = 1, hidden2_size do
                    delta_hidden2[j] = 0
                    for k = 1, hidden3_size do
                        delta_hidden2[j] = delta_hidden2[j]
                            + delta_hidden3[k] * weights_hidden2_hidden3[j][k] * (1 - hidden2_input[j])
                    end
                end

                delta_hidden1 = {}
                for j = 1, hidden1_size do
                    delta_hidden1[j] = 0
                    for k = 1, hidden2_size do
                        delta_hidden1[j] = delta_hidden1[j]
                            + delta_hidden2[k] * weights_hidden1_hidden2[j][k] * (1 - hidden1_input[j])
                    end
                end

                -- Aktualizacja wag zgodnie z AdamW
                for j = 1, hidden3_size do
                    m_hidden3_output[j] = beta1 * m_hidden3_output[j] + (1 - beta1) * delta_output * hidden3_output[j]
                    v_hidden3_output[j] = beta2 * v_hidden3_output[j]
                        + (1 - beta2) * (delta_output * hidden3_output[j]) ^ 2
                    m_hat = m_hidden3_output[j] / (1 - beta1 ^ epoch)
                    v_hat = v_hidden3_output[j] / (1 - beta2 ^ epoch)
                    wd_hidden3_output[j] = weight_decay * weights_hidden3_output[j]
                    weights_hidden3_output[j] = weights_hidden3_output[j]
                        + learning_rate * (m_hat / (math.sqrt(v_hat) + epsilon) - wd_hidden3_output[j])
                end

                for j = 1, hidden2_size do
                    for k = 1, hidden3_size do
                        m_hidden2_hidden3[j][k] = beta1 * m_hidden2_hidden3[j][k]
                            + (1 - beta1) * delta_hidden3[k] * hidden2_input[j]
                        v_hidden2_hidden3[j][k] = beta2 * v_hidden2_hidden3[j][k]
                            + (1 - beta2) * (delta_hidden3[k] * hidden2_input[j]) ^ 2
                        m_hat = m_hidden2_hidden3[j][k] / (1 - beta1 ^ epoch)
                        v_hat = v_hidden2_hidden3[j][k] / (1 - beta2 ^ epoch)
                        wd_hidden2_hidden3[j][k] = weight_decay * weights_hidden2_hidden3[j][k]
                        weights_hidden2_hidden3[j][k] = weights_hidden2_hidden3[j][k]
                            + learning_rate * (m_hat / (math.sqrt(v_hat) + epsilon) - wd_hidden2_hidden3[j][k])
                    end
                end

                for j = 1, hidden1_size do
                    for k = 1, hidden2_size do
                        m_hidden1_hidden2[j][k] = beta1 * m_hidden1_hidden2[j][k]
                            + (1 - beta1) * delta_hidden2[k] * hidden1_input[j]
                        v_hidden1_hidden2[j][k] = beta2 * v_hidden1_hidden2[j][k]
                            + (1 - beta2) * (delta_hidden2[k] * hidden1_input[j]) ^ 2
                        m_hat = m_hidden1_hidden2[j][k] / (1 - beta1 ^ epoch)
                        v_hat = v_hidden1_hidden2[j][k] / (1 - beta2 ^ epoch)
                        wd_hidden1_hidden2[j][k] = weight_decay * weights_hidden1_hidden2[j][k]
                        weights_hidden1_hidden2[j][k] = weights_hidden1_hidden2[j][k]
                            + learning_rate * (m_hat / (math.sqrt(v_hat) + epsilon) - wd_hidden1_hidden2[j][k])
                    end
                end

                for j = 1, input_size do
                    for k = 1, hidden1_size do
                        m_input_hidden1[j][k] = beta1 * m_input_hidden1[j][k]
                            + (1 - beta1) * delta_hidden1[k] * x_train[i][j]
                        v_input_hidden1[j][k] = beta2 * v_input_hidden1[j][k]
                            + (1 - beta2) * (delta_hidden1[k] * x_train[i][j]) ^ 2
                        m_hat = m_input_hidden1[j][k] / (1 - beta1 ^ epoch)
                        v_hat = v_input_hidden1[j][k] / (1 - beta2 ^ epoch)
                        wd_input_hidden1[j][k] = weight_decay * weights_input_hidden1[j][k]
                        weights_input_hidden1[j][k] = weights_input_hidden1[j][k]
                            + learning_rate * (m_hat / (math.sqrt(v_hat) + epsilon) - wd_input_hidden1[j][k])
                    end
                end
            end
        end

        -- Testowanie sieci
        for i = 1, #x_test do
            print("")

            hidden1_input = {}
            hidden2_input = {}
            hidden3_input = {}
            hidden3_output = {}
            final_input = {}

            for j = 1, hidden1_size do
                hidden1_input[j] = 0
                for k = 1, input_size do
                    hidden1_input[j] = hidden1_input[j] + x_test[i][k] * weights_input_hidden1[k][j]
                end
                hidden1_input[j] = activation_input(hidden1_input[j])
            end

            for j = 1, hidden2_size do
                hidden2_input[j] = 0
                for k = 1, hidden1_size do
                    hidden2_input[j] = hidden2_input[j] + hidden1_input[k] * weights_hidden1_hidden2[k][j]
                end
                hidden2_input[j] = activation_hd(hidden2_input[j])
            end

            for j = 1, hidden3_size do
                hidden3_input[j] = 0
                for k = 1, hidden2_size do
                    hidden3_input[j] = hidden3_input[j] + hidden2_input[k] * weights_hidden2_hidden3[k][j]
                end
                hidden3_input[j] = activation_hd(hidden3_input[j])
            end

            for j = 1, hidden3_size do
                hidden3_output[j] = activation_hd(hidden3_input[j])
            end

            final_input[1] = 0
            for j = 1, hidden3_size do
                final_input[1] = final_input[1] + hidden3_output[j] * weights_hidden3_output[j]
            end
            final_output = activation_output(final_input[1])

            print("Test input:      " .. grey, table.concat(x_test[i], " ") .. reset)
            print("Predicted output:" .. red, final_output .. reset)
        end
    end

    if hidden_layers == 4 then
        weights_input_hidden1 = {}
        weights_hidden1_hidden2 = {}
        weights_hidden2_hidden3 = {}
        weights_hidden3_hidden4 = {}
        weights_hidden4_output = {}

        for i = 1, input_size do
            weights_input_hidden1[i] = {}
            for j = 1, hidden1_size do
                weights_input_hidden1[i][j] = math.random()
            end
        end

        for i = 1, hidden1_size do
            weights_hidden1_hidden2[i] = {}
            for j = 1, hidden2_size do
                weights_hidden1_hidden2[i][j] = math.random()
            end
        end

        for i = 1, hidden2_size do
            weights_hidden2_hidden3[i] = {}
            for j = 1, hidden3_size do
                weights_hidden2_hidden3[i][j] = math.random()
            end
        end

        for i = 1, hidden3_size do
            weights_hidden3_hidden4[i] = {}
            for j = 1, hidden4_size do
                weights_hidden3_hidden4[i][j] = math.random()
            end
        end

        for i = 1, hidden4_size do
            weights_hidden4_output[i] = math.random()
        end

        -- Inicjalizacja zmiennych AdamW
        m_input_hidden1 = {}
        v_input_hidden1 = {}
        m_hidden1_hidden2 = {}
        v_hidden1_hidden2 = {}
        m_hidden2_hidden3 = {}
        v_hidden2_hidden3 = {}
        m_hidden3_hidden4 = {}
        v_hidden3_hidden4 = {}
        m_hidden4_output = {}
        v_hidden4_output = {}
        wd_input_hidden1 = {}
        wd_hidden1_hidden2 = {}
        wd_hidden2_hidden3 = {}
        wd_hidden3_hidden4 = {}
        wd_hidden4_output = {}

        for i = 1, input_size do
            m_input_hidden1[i] = {}
            v_input_hidden1[i] = {}
            wd_input_hidden1[i] = {}
            for j = 1, hidden1_size do
                m_input_hidden1[i][j] = 0
                v_input_hidden1[i][j] = 0
                wd_input_hidden1[i][j] = 0
            end
        end

        for i = 1, hidden1_size do
            m_hidden1_hidden2[i] = {}
            v_hidden1_hidden2[i] = {}
            wd_hidden1_hidden2[i] = {}
            for j = 1, hidden2_size do
                m_hidden1_hidden2[i][j] = 0
                v_hidden1_hidden2[i][j] = 0
                wd_hidden1_hidden2[i][j] = 0
            end
        end

        for i = 1, hidden2_size do
            m_hidden2_hidden3[i] = {}
            v_hidden2_hidden3[i] = {}
            wd_hidden2_hidden3[i] = {}
            for j = 1, hidden3_size do
                m_hidden2_hidden3[i][j] = 0
                v_hidden2_hidden3[i][j] = 0
                wd_hidden2_hidden3[i][j] = 0
            end
        end

        for i = 1, hidden3_size do
            m_hidden3_hidden4[i] = {}
            v_hidden3_hidden4[i] = {}
            wd_hidden3_hidden4[i] = {}
            for j = 1, hidden4_size do
                m_hidden3_hidden4[i][j] = 0
                v_hidden3_hidden4[i][j] = 0
                wd_hidden3_hidden4[i][j] = 0
            end
        end

        for i = 1, hidden4_size do
            m_hidden4_output[i] = 0
            v_hidden4_output[i] = 0
            wd_hidden4_output[i] = 0
        end

        -- Uczenie sieci
        epochs = iterations
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        learning_rate = 0.001

        for epoch = 1, epochs do
            for i = 1, #x_train do
                -- Forward pass
                hidden1_input = {}
                hidden2_input = {}
                hidden3_input = {}
                hidden4_input = {}
                hidden4_output = {}
                final_input = {}

                for j = 1, hidden1_size do
                    hidden1_input[j] = 0
                    for k = 1, input_size do
                        hidden1_input[j] = hidden1_input[j] + x_train[i][k] * weights_input_hidden1[k][j]
                    end
                    hidden1_input[j] = activation_input(hidden1_input[j])
                end

                for j = 1, hidden2_size do
                    hidden2_input[j] = 0
                    for k = 1, hidden1_size do
                        hidden2_input[j] = hidden2_input[j] + hidden1_input[k] * weights_hidden1_hidden2[k][j]
                    end
                    hidden2_input[j] = activation_hd(hidden2_input[j])
                end

                for j = 1, hidden3_size do
                    hidden3_input[j] = 0
                    for k = 1, hidden2_size do
                        hidden3_input[j] = hidden3_input[j] + hidden2_input[k] * weights_hidden2_hidden3[k][j]
                    end
                    hidden3_input[j] = activation_hd(hidden3_input[j])
                end

                for j = 1, hidden4_size do
                    hidden4_input[j] = 0
                    for k = 1, hidden3_size do
                        hidden4_input[j] = hidden4_input[j] + hidden3_input[k] * weights_hidden3_hidden4[k][j]
                    end
                    hidden4_input[j] = activation_hd(hidden4_input[j])
                end

                for j = 1, hidden4_size do
                    hidden4_output[j] = activation_hd(hidden4_input[j])
                end

                final_input[1] = 0
                for j = 1, hidden4_size do
                    final_input[1] = final_input[1] + hidden4_output[j] * weights_hidden4_output[j]
                end
                final_output = activation_output(final_input[1])

                -- Obliczenie błędu
                error = y_train[i] - final_output

                -- Backpropagation
                delta_output = error * final_output * (1 - final_output)

                delta_hidden4 = {}
                for j = 1, hidden4_size do
                    delta_hidden4[j] = delta_output * weights_hidden4_output[j] * (1 - hidden4_output[j])
                end

                delta_hidden3 = {}
                for j = 1, hidden3_size do
                    delta_hidden3[j] = 0
                    for k = 1, hidden4_size do
                        delta_hidden3[j] = delta_hidden3[j]
                            + delta_hidden4[k] * weights_hidden3_hidden4[j][k] * (1 - hidden3_input[j])
                    end
                end

                delta_hidden2 = {}
                for j = 1, hidden2_size do
                    delta_hidden2[j] = 0
                    for k = 1, hidden3_size do
                        delta_hidden2[j] = delta_hidden2[j]
                            + delta_hidden3[k] * weights_hidden2_hidden3[j][k] * (1 - hidden2_input[j])
                    end
                end

                delta_hidden1 = {}
                for j = 1, hidden1_size do
                    delta_hidden1[j] = 0
                    for k = 1, hidden2_size do
                        delta_hidden1[j] = delta_hidden1[j]
                            + delta_hidden2[k] * weights_hidden1_hidden2[j][k] * (1 - hidden1_input[j])
                    end
                end

                -- Aktualizacja wag zgodnie z AdamW
                for j = 1, hidden4_size do
                    m_hidden4_output[j] = beta1 * m_hidden4_output[j] + (1 - beta1) * delta_output * hidden4_output[j]
                    v_hidden4_output[j] = beta2 * v_hidden4_output[j]
                        + (1 - beta2) * (delta_output * hidden4_output[j]) ^ 2
                    m_hat = m_hidden4_output[j] / (1 - beta1 ^ epoch)
                    v_hat = v_hidden4_output[j] / (1 - beta2 ^ epoch)
                    wd_hidden4_output[j] = weight_decay * weights_hidden4_output[j]
                    weights_hidden4_output[j] = weights_hidden4_output[j]
                        + learning_rate * (m_hat / (math.sqrt(v_hat) + epsilon) - wd_hidden4_output[j])
                end

                for j = 1, hidden3_size do
                    for k = 1, hidden4_size do
                        m_hidden3_hidden4[j][k] = beta1 * m_hidden3_hidden4[j][k]
                            + (1 - beta1) * delta_hidden4[k] * hidden3_input[j]
                        v_hidden3_hidden4[j][k] = beta2 * v_hidden3_hidden4[j][k]
                            + (1 - beta2) * (delta_hidden4[k] * hidden3_input[j]) ^ 2
                        m_hat = m_hidden3_hidden4[j][k] / (1 - beta1 ^ epoch)
                        v_hat = v_hidden3_hidden4[j][k] / (1 - beta2 ^ epoch)
                        wd_hidden3_hidden4[j][k] = weight_decay * weights_hidden3_hidden4[j][k]
                        weights_hidden3_hidden4[j][k] = weights_hidden3_hidden4[j][k]
                            + learning_rate * (m_hat / (math.sqrt(v_hat) + epsilon) - wd_hidden3_hidden4[j][k])
                    end
                end

                for j = 1, hidden2_size do
                    for k = 1, hidden3_size do
                        m_hidden2_hidden3[j][k] = beta1 * m_hidden2_hidden3[j][k]
                            + (1 - beta1) * delta_hidden3[k] * hidden2_input[j]
                        v_hidden2_hidden3[j][k] = beta2 * v_hidden2_hidden3[j][k]
                            + (1 - beta2) * (delta_hidden3[k] * hidden2_input[j]) ^ 2
                        m_hat = m_hidden2_hidden3[j][k] / (1 - beta1 ^ epoch)
                        v_hat = v_hidden2_hidden3[j][k] / (1 - beta2 ^ epoch)
                        wd_hidden2_hidden3[j][k] = weight_decay * weights_hidden2_hidden3[j][k]
                        weights_hidden2_hidden3[j][k] = weights_hidden2_hidden3[j][k]
                            + learning_rate * (m_hat / (math.sqrt(v_hat) + epsilon) - wd_hidden2_hidden3[j][k])
                    end
                end

                for j = 1, hidden1_size do
                    for k = 1, hidden2_size do
                        m_hidden1_hidden2[j][k] = beta1 * m_hidden1_hidden2[j][k]
                            + (1 - beta1) * delta_hidden2[k] * hidden1_input[j]
                        v_hidden1_hidden2[j][k] = beta2 * v_hidden1_hidden2[j][k]
                            + (1 - beta2) * (delta_hidden2[k] * hidden1_input[j]) ^ 2
                        m_hat = m_hidden1_hidden2[j][k] / (1 - beta1 ^ epoch)
                        v_hat = v_hidden1_hidden2[j][k] / (1 - beta2 ^ epoch)
                        wd_hidden1_hidden2[j][k] = weight_decay * weights_hidden1_hidden2[j][k]
                        weights_hidden1_hidden2[j][k] = weights_hidden1_hidden2[j][k]
                            + learning_rate * (m_hat / (math.sqrt(v_hat) + epsilon) - wd_hidden1_hidden2[j][k])
                    end
                end

                for j = 1, input_size do
                    for k = 1, hidden1_size do
                        m_input_hidden1[j][k] = beta1 * m_input_hidden1[j][k]
                            + (1 - beta1) * delta_hidden1[k] * x_train[i][j]
                        v_input_hidden1[j][k] = beta2 * v_input_hidden1[j][k]
                            + (1 - beta2) * (delta_hidden1[k] * x_train[i][j]) ^ 2
                        m_hat = m_input_hidden1[j][k] / (1 - beta1 ^ epoch)
                        v_hat = v_input_hidden1[j][k] / (1 - beta2 ^ epoch)
                        wd_input_hidden1[j][k] = weight_decay * weights_input_hidden1[j][k]
                        weights_input_hidden1[j][k] = weights_input_hidden1[j][k]
                            + learning_rate * (m_hat / (math.sqrt(v_hat) + epsilon) - wd_input_hidden1[j][k])
                    end
                end
            end
        end

        -- Testowanie sieci
        for i = 1, #x_test do
            print("")

            hidden1_input = {}
            hidden2_input = {}
            hidden3_input = {}
            hidden4_input = {}
            hidden4_output = {}
            final_input = {}

            for j = 1, hidden1_size do
                hidden1_input[j] = 0
                for k = 1, input_size do
                    hidden1_input[j] = hidden1_input[j] + x_test[i][k] * weights_input_hidden1[k][j]
                end
                hidden1_input[j] = activation_input(hidden1_input[j])
            end

            for j = 1, hidden2_size do
                hidden2_input[j] = 0
                for k = 1, hidden1_size do
                    hidden2_input[j] = hidden2_input[j] + hidden1_input[k] * weights_hidden1_hidden2[k][j]
                end
                hidden2_input[j] = activation_hd(hidden2_input[j])
            end

            for j = 1, hidden3_size do
                hidden3_input[j] = 0
                for k = 1, hidden2_size do
                    hidden3_input[j] = hidden3_input[j] + hidden2_input[k] * weights_hidden2_hidden3[k][j]
                end
                hidden3_input[j] = activation_hd(hidden3_input[j])
            end

            for j = 1, hidden4_size do
                hidden4_input[j] = 0
                for k = 1, hidden3_size do
                    hidden4_input[j] = hidden4_input[j] + hidden3_input[k] * weights_hidden3_hidden4[k][j]
                end
                hidden4_input[j] = activation_hd(hidden4_input[j])
            end

            for j = 1, hidden4_size do
                hidden4_output[j] = activation_hd(hidden4_input[j])
            end

            final_input[1] = 0
            for j = 1, hidden4_size do
                final_input[1] = final_input[1] + hidden4_output[j] * weights_hidden4_output[j]
            end
            final_output = activation_output(final_input[1])

            print("Test input:      " .. grey, table.concat(x_test[i], " ") .. reset)
            print("Predicted output:" .. red, final_output .. reset)
        end
    end

    if hidden_layers == 5 then
        weights_input_hidden1 = {}
        weights_hidden1_hidden2 = {}
        weights_hidden2_hidden3 = {}
        weights_hidden3_hidden4 = {}
        weights_hidden4_hidden5 = {}
        weights_hidden5_output = {}

        for i = 1, input_size do
            weights_input_hidden1[i] = {}
            for j = 1, hidden1_size do
                weights_input_hidden1[i][j] = math.random()
            end
        end

        for i = 1, hidden1_size do
            weights_hidden1_hidden2[i] = {}
            for j = 1, hidden2_size do
                weights_hidden1_hidden2[i][j] = math.random()
            end
        end

        for i = 1, hidden2_size do
            weights_hidden2_hidden3[i] = {}
            for j = 1, hidden3_size do
                weights_hidden2_hidden3[i][j] = math.random()
            end
        end

        for i = 1, hidden3_size do
            weights_hidden3_hidden4[i] = {}
            for j = 1, hidden4_size do
                weights_hidden3_hidden4[i][j] = math.random()
            end
        end

        for i = 1, hidden4_size do
            weights_hidden4_hidden5[i] = {}
            for j = 1, hidden5_size do
                weights_hidden4_hidden5[i][j] = math.random()
            end
        end

        for i = 1, hidden5_size do
            weights_hidden5_output[i] = math.random()
        end

        -- Inicjalizacja zmiennych AdamW
        m_input_hidden1 = {}
        v_input_hidden1 = {}
        m_hidden1_hidden2 = {}
        v_hidden1_hidden2 = {}
        m_hidden2_hidden3 = {}
        v_hidden2_hidden3 = {}
        m_hidden3_hidden4 = {}
        v_hidden3_hidden4 = {}
        m_hidden4_hidden5 = {}
        v_hidden4_hidden5 = {}
        m_hidden5_output = {}
        v_hidden5_output = {}
        wd_input_hidden1 = {}
        wd_hidden1_hidden2 = {}
        wd_hidden2_hidden3 = {}
        wd_hidden3_hidden4 = {}
        wd_hidden4_hidden5 = {}
        wd_hidden5_output = {}

        for i = 1, input_size do
            m_input_hidden1[i] = {}
            v_input_hidden1[i] = {}
            wd_input_hidden1[i] = {}
            for j = 1, hidden1_size do
                m_input_hidden1[i][j] = 0
                v_input_hidden1[i][j] = 0
                wd_input_hidden1[i][j] = 0
            end
        end

        for i = 1, hidden1_size do
            m_hidden1_hidden2[i] = {}
            v_hidden1_hidden2[i] = {}
            wd_hidden1_hidden2[i] = {}
            for j = 1, hidden2_size do
                m_hidden1_hidden2[i][j] = 0
                v_hidden1_hidden2[i][j] = 0
                wd_hidden1_hidden2[i][j] = 0
            end
        end

        for i = 1, hidden2_size do
            m_hidden2_hidden3[i] = {}
            v_hidden2_hidden3[i] = {}
            wd_hidden2_hidden3[i] = {}
            for j = 1, hidden3_size do
                m_hidden2_hidden3[i][j] = 0
                v_hidden2_hidden3[i][j] = 0
                wd_hidden2_hidden3[i][j] = 0
            end
        end

        for i = 1, hidden3_size do
            m_hidden3_hidden4[i] = {}
            v_hidden3_hidden4[i] = {}
            wd_hidden3_hidden4[i] = {}
            for j = 1, hidden4_size do
                m_hidden3_hidden4[i][j] = 0
                v_hidden3_hidden4[i][j] = 0
                wd_hidden3_hidden4[i][j] = 0
            end
        end

        for i = 1, hidden4_size do
            m_hidden4_hidden5[i] = {}
            v_hidden4_hidden5[i] = {}
            wd_hidden4_hidden5[i] = {}
            for j = 1, hidden5_size do
                m_hidden4_hidden5[i][j] = 0
                v_hidden4_hidden5[i][j] = 0
                wd_hidden4_hidden5[i][j] = 0
            end
        end

        for i = 1, hidden5_size do
            m_hidden5_output[i] = 0
            v_hidden5_output[i] = 0
            wd_hidden5_output[i] = 0
        end

        -- Uczenie sieci
        epochs = iterations
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        learning_rate = 0.001

        for epoch = 1, epochs do
            for i = 1, #x_train do
                -- Forward pass
                hidden1_input = {}
                hidden2_input = {}
                hidden3_input = {}
                hidden4_input = {}
                hidden5_input = {}
                hidden5_output = {}
                final_input = {}

                for j = 1, hidden1_size do
                    hidden1_input[j] = 0
                    for k = 1, input_size do
                        hidden1_input[j] = hidden1_input[j] + x_train[i][k] * weights_input_hidden1[k][j]
                    end
                    hidden1_input[j] = activation_input(hidden1_input[j])
                end

                for j = 1, hidden2_size do
                    hidden2_input[j] = 0
                    for k = 1, hidden1_size do
                        hidden2_input[j] = hidden2_input[j] + hidden1_input[k] * weights_hidden1_hidden2[k][j]
                    end
                    hidden2_input[j] = activation_hd(hidden2_input[j])
                end

                for j = 1, hidden3_size do
                    hidden3_input[j] = 0
                    for k = 1, hidden2_size do
                        hidden3_input[j] = hidden3_input[j] + hidden2_input[k] * weights_hidden2_hidden3[k][j]
                    end
                    hidden3_input[j] = activation_hd(hidden3_input[j])
                end

                for j = 1, hidden4_size do
                    hidden4_input[j] = 0
                    for k = 1, hidden3_size do
                        hidden4_input[j] = hidden4_input[j] + hidden3_input[k] * weights_hidden3_hidden4[k][j]
                    end
                    hidden4_input[j] = activation_hd(hidden4_input[j])
                end

                for j = 1, hidden5_size do
                    hidden5_input[j] = 0
                    for k = 1, hidden4_size do
                        hidden5_input[j] = hidden5_input[j] + hidden4_input[k] * weights_hidden4_hidden5[k][j]
                    end
                    hidden5_input[j] = activation_hd(hidden5_input[j])
                end

                for j = 1, hidden5_size do
                    hidden5_output[j] = activation_hd(hidden5_input[j])
                end

                final_input[1] = 0
                for j = 1, hidden5_size do
                    final_input[1] = final_input[1] + hidden5_output[j] * weights_hidden5_output[j]
                end
                final_output = activation_output(final_input[1])

                -- Obliczenie błędu
                error = y_train[i] - final_output

                -- Backpropagation
                delta_output = error * final_output * (1 - final_output)

                delta_hidden5 = {}
                for j = 1, hidden5_size do
                    delta_hidden5[j] = delta_output * weights_hidden5_output[j] * (1 - hidden5_output[j])
                end

                delta_hidden4 = {}
                for j = 1, hidden4_size do
                    delta_hidden4[j] = 0
                    for k = 1, hidden5_size do
                        delta_hidden4[j] = delta_hidden4[j]
                            + delta_hidden5[k] * weights_hidden4_hidden5[j][k] * (1 - hidden4_input[j])
                    end
                end

                delta_hidden3 = {}
                for j = 1, hidden3_size do
                    delta_hidden3[j] = 0
                    for k = 1, hidden4_size do
                        delta_hidden3[j] = delta_hidden3[j]
                            + delta_hidden4[k] * weights_hidden3_hidden4[j][k] * (1 - hidden3_input[j])
                    end
                end

                delta_hidden2 = {}
                for j = 1, hidden2_size do
                    delta_hidden2[j] = 0
                    for k = 1, hidden3_size do
                        delta_hidden2[j] = delta_hidden2[j]
                            + delta_hidden3[k] * weights_hidden2_hidden3[j][k] * (1 - hidden2_input[j])
                    end
                end

                delta_hidden1 = {}
                for j = 1, hidden1_size do
                    delta_hidden1[j] = 0
                    for k = 1, hidden2_size do
                        delta_hidden1[j] = delta_hidden1[j]
                            + delta_hidden2[k] * weights_hidden1_hidden2[j][k] * (1 - hidden1_input[j])
                    end
                end

                -- Aktualizacja wag zgodnie z AdamW
                for j = 1, hidden5_size do
                    m_hidden5_output[j] = beta1 * m_hidden5_output[j] + (1 - beta1) * delta_output * hidden5_output[j]
                    v_hidden5_output[j] = beta2 * v_hidden5_output[j]
                        + (1 - beta2) * (delta_output * hidden5_output[j]) ^ 2
                    m_hat = m_hidden5_output[j] / (1 - beta1 ^ epoch)
                    v_hat = v_hidden5_output[j] / (1 - beta2 ^ epoch)
                    wd_hidden5_output[j] = weight_decay * weights_hidden5_output[j]
                    weights_hidden5_output[j] = weights_hidden5_output[j]
                        + learning_rate * (m_hat / (math.sqrt(v_hat) + epsilon) - wd_hidden5_output[j])
                end

                for j = 1, hidden4_size do
                    for k = 1, hidden5_size do
                        m_hidden4_hidden5[j][k] = beta1 * m_hidden4_hidden5[j][k]
                            + (1 - beta1) * delta_hidden5[k] * hidden4_input[j]
                        v_hidden4_hidden5[j][k] = beta2 * v_hidden4_hidden5[j][k]
                            + (1 - beta2) * (delta_hidden5[k] * hidden4_input[j]) ^ 2
                        m_hat = m_hidden4_hidden5[j][k] / (1 - beta1 ^ epoch)
                        v_hat = v_hidden4_hidden5[j][k] / (1 - beta2 ^ epoch)
                        wd_hidden4_hidden5[j][k] = weight_decay * weights_hidden4_hidden5[j][k]
                        weights_hidden4_hidden5[j][k] = weights_hidden4_hidden5[j][k]
                            + learning_rate * (m_hat / (math.sqrt(v_hat) + epsilon) - wd_hidden4_hidden5[j][k])
                    end
                end

                for j = 1, hidden3_size do
                    for k = 1, hidden4_size do
                        m_hidden3_hidden4[j][k] = beta1 * m_hidden3_hidden4[j][k]
                            + (1 - beta1) * delta_hidden4[k] * hidden3_input[j]
                        v_hidden3_hidden4[j][k] = beta2 * v_hidden3_hidden4[j][k]
                            + (1 - beta2) * (delta_hidden4[k] * hidden3_input[j]) ^ 2
                        m_hat = m_hidden3_hidden4[j][k] / (1 - beta1 ^ epoch)
                        v_hat = v_hidden3_hidden4[j][k] / (1 - beta2 ^ epoch)
                        wd_hidden3_hidden4[j][k] = weight_decay * weights_hidden3_hidden4[j][k]
                        weights_hidden3_hidden4[j][k] = weights_hidden3_hidden4[j][k]
                            + learning_rate * (m_hat / (math.sqrt(v_hat) + epsilon) - wd_hidden3_hidden4[j][k])
                    end
                end

                for j = 1, hidden2_size do
                    for k = 1, hidden3_size do
                        m_hidden2_hidden3[j][k] = beta1 * m_hidden2_hidden3[j][k]
                            + (1 - beta1) * delta_hidden3[k] * hidden2_input[j]
                        v_hidden2_hidden3[j][k] = beta2 * v_hidden2_hidden3[j][k]
                            + (1 - beta2) * (delta_hidden3[k] * hidden2_input[j]) ^ 2
                        m_hat = m_hidden2_hidden3[j][k] / (1 - beta1 ^ epoch)
                        v_hat = v_hidden2_hidden3[j][k] / (1 - beta2 ^ epoch)
                        wd_hidden2_hidden3[j][k] = weight_decay * weights_hidden2_hidden3[j][k]
                        weights_hidden2_hidden3[j][k] = weights_hidden2_hidden3[j][k]
                            + learning_rate * (m_hat / (math.sqrt(v_hat) + epsilon) - wd_hidden2_hidden3[j][k])
                    end
                end

                for j = 1, hidden1_size do
                    for k = 1, hidden2_size do
                        m_hidden1_hidden2[j][k] = beta1 * m_hidden1_hidden2[j][k]
                            + (1 - beta1) * delta_hidden2[k] * hidden1_input[j]
                        v_hidden1_hidden2[j][k] = beta2 * v_hidden1_hidden2[j][k]
                            + (1 - beta2) * (delta_hidden2[k] * hidden1_input[j]) ^ 2
                        m_hat = m_hidden1_hidden2[j][k] / (1 - beta1 ^ epoch)
                        v_hat = v_hidden1_hidden2[j][k] / (1 - beta2 ^ epoch)
                        wd_hidden1_hidden2[j][k] = weight_decay * weights_hidden1_hidden2[j][k]
                        weights_hidden1_hidden2[j][k] = weights_hidden1_hidden2[j][k]
                            + learning_rate * (m_hat / (math.sqrt(v_hat) + epsilon) - wd_hidden1_hidden2[j][k])
                    end
                end

                for j = 1, input_size do
                    for k = 1, hidden1_size do
                        m_input_hidden1[j][k] = beta1 * m_input_hidden1[j][k]
                            + (1 - beta1) * delta_hidden1[k] * x_train[i][j]
                        v_input_hidden1[j][k] = beta2 * v_input_hidden1[j][k]
                            + (1 - beta2) * (delta_hidden1[k] * x_train[i][j]) ^ 2
                        m_hat = m_input_hidden1[j][k] / (1 - beta1 ^ epoch)
                        v_hat = v_input_hidden1[j][k] / (1 - beta2 ^ epoch)
                        wd_input_hidden1[j][k] = weight_decay * weights_input_hidden1[j][k]
                        weights_input_hidden1[j][k] = weights_input_hidden1[j][k]
                            + learning_rate * (m_hat / (math.sqrt(v_hat) + epsilon) - wd_input_hidden1[j][k])
                    end
                end
            end
        end

        -- Testowanie sieci
        for i = 1, #x_test do
            print("")

            hidden1_input = {}
            hidden2_input = {}
            hidden3_input = {}
            hidden4_input = {}
            hidden5_input = {}
            hidden5_output = {}
            final_input = {}

            for j = 1, hidden1_size do
                hidden1_input[j] = 0
                for k = 1, input_size do
                    hidden1_input[j] = hidden1_input[j] + x_test[i][k] * weights_input_hidden1[k][j]
                end
                hidden1_input[j] = activation_input(hidden1_input[j])
            end

            for j = 1, hidden2_size do
                hidden2_input[j] = 0
                for k = 1, hidden1_size do
                    hidden2_input[j] = hidden2_input[j] + hidden1_input[k] * weights_hidden1_hidden2[k][j]
                end
                hidden2_input[j] = activation_hd(hidden2_input[j])
            end

            for j = 1, hidden3_size do
                hidden3_input[j] = 0
                for k = 1, hidden2_size do
                    hidden3_input[j] = hidden3_input[j] + hidden2_input[k] * weights_hidden2_hidden3[k][j]
                end
                hidden3_input[j] = activation_hd(hidden3_input[j])
            end

            for j = 1, hidden4_size do
                hidden4_input[j] = 0
                for k = 1, hidden3_size do
                    hidden4_input[j] = hidden4_input[j] + hidden3_input[k] * weights_hidden3_hidden4[k][j]
                end
                hidden4_input[j] = activation_hd(hidden4_input[j])
            end

            for j = 1, hidden5_size do
                hidden5_input[j] = 0
                for k = 1, hidden4_size do
                    hidden5_input[j] = hidden5_input[j] + hidden4_input[k] * weights_hidden4_hidden5[k][j]
                end
                hidden5_input[j] = activation_hd(hidden5_input[j])
            end

            for j = 1, hidden5_size do
                hidden5_output[j] = activation_hd(hidden5_input[j])
            end

            final_input[1] = 0
            for j = 1, hidden5_size do
                final_input[1] = final_input[1] + hidden5_output[j] * weights_hidden5_output[j]
            end
            final_output = activation_output(final_input[1])

            print("Test input:      " .. grey, table.concat(x_test[i], " ") .. reset)
            print("Predicted output:" .. red, final_output .. reset)
        end
    end

    if hidden_layers == 6 then
        weights_input_hidden1 = {}
        weights_hidden1_hidden2 = {}
        weights_hidden2_hidden3 = {}
        weights_hidden3_hidden4 = {}
        weights_hidden4_hidden5 = {}
        weights_hidden5_hidden6 = {}
        weights_hidden6_output = {}

        for i = 1, input_size do
            weights_input_hidden1[i] = {}
            for j = 1, hidden1_size do
                weights_input_hidden1[i][j] = math.random()
            end
        end

        for i = 1, hidden1_size do
            weights_hidden1_hidden2[i] = {}
            for j = 1, hidden2_size do
                weights_hidden1_hidden2[i][j] = math.random()
            end
        end

        for i = 1, hidden2_size do
            weights_hidden2_hidden3[i] = {}
            for j = 1, hidden3_size do
                weights_hidden2_hidden3[i][j] = math.random()
            end
        end

        for i = 1, hidden3_size do
            weights_hidden3_hidden4[i] = {}
            for j = 1, hidden4_size do
                weights_hidden3_hidden4[i][j] = math.random()
            end
        end

        for i = 1, hidden4_size do
            weights_hidden4_hidden5[i] = {}
            for j = 1, hidden5_size do
                weights_hidden4_hidden5[i][j] = math.random()
            end
        end

        for i = 1, hidden5_size do
            weights_hidden5_hidden6[i] = {}
            for j = 1, hidden6_size do
                weights_hidden5_hidden6[i][j] = math.random()
            end
        end

        for i = 1, hidden6_size do
            weights_hidden6_output[i] = math.random()
        end

        -- Inicjalizacja zmiennych AdamW
        m_input_hidden1 = {}
        v_input_hidden1 = {}
        m_hidden1_hidden2 = {}
        v_hidden1_hidden2 = {}
        m_hidden2_hidden3 = {}
        v_hidden2_hidden3 = {}
        m_hidden3_hidden4 = {}
        v_hidden3_hidden4 = {}
        m_hidden4_hidden5 = {}
        v_hidden4_hidden5 = {}
        m_hidden5_hidden6 = {}
        v_hidden5_hidden6 = {}
        m_hidden6_output = {}
        v_hidden6_output = {}
        wd_input_hidden1 = {}
        wd_hidden1_hidden2 = {}
        wd_hidden2_hidden3 = {}
        wd_hidden3_hidden4 = {}
        wd_hidden4_hidden5 = {}
        wd_hidden5_hidden6 = {}
        wd_hidden6_output = {}

        for i = 1, input_size do
            m_input_hidden1[i] = {}
            v_input_hidden1[i] = {}
            wd_input_hidden1[i] = {}
            for j = 1, hidden1_size do
                m_input_hidden1[i][j] = 0
                v_input_hidden1[i][j] = 0
                wd_input_hidden1[i][j] = 0
            end
        end

        for i = 1, hidden1_size do
            m_hidden1_hidden2[i] = {}
            v_hidden1_hidden2[i] = {}
            wd_hidden1_hidden2[i] = {}
            for j = 1, hidden2_size do
                m_hidden1_hidden2[i][j] = 0
                v_hidden1_hidden2[i][j] = 0
                wd_hidden1_hidden2[i][j] = 0
            end
        end

        for i = 1, hidden2_size do
            m_hidden2_hidden3[i] = {}
            v_hidden2_hidden3[i] = {}
            wd_hidden2_hidden3[i] = {}
            for j = 1, hidden3_size do
                m_hidden2_hidden3[i][j] = 0
                v_hidden2_hidden3[i][j] = 0
                wd_hidden2_hidden3[i][j] = 0
            end
        end

        for i = 1, hidden3_size do
            m_hidden3_hidden4[i] = {}
            v_hidden3_hidden4[i] = {}
            wd_hidden3_hidden4[i] = {}
            for j = 1, hidden4_size do
                m_hidden3_hidden4[i][j] = 0
                v_hidden3_hidden4[i][j] = 0
                wd_hidden3_hidden4[i][j] = 0
            end
        end

        for i = 1, hidden4_size do
            m_hidden4_hidden5[i] = {}
            v_hidden4_hidden5[i] = {}
            wd_hidden4_hidden5[i] = {}
            for j = 1, hidden5_size do
                m_hidden4_hidden5[i][j] = 0
                v_hidden4_hidden5[i][j] = 0
                wd_hidden4_hidden5[i][j] = 0
            end
        end

        for i = 1, hidden5_size do
            m_hidden5_hidden6[i] = {}
            v_hidden5_hidden6[i] = {}
            wd_hidden5_hidden6[i] = {}
            for j = 1, hidden6_size do
                m_hidden5_hidden6[i][j] = 0
                v_hidden5_hidden6[i][j] = 0
                wd_hidden5_hidden6[i][j] = 0
            end
        end

        for i = 1, hidden6_size do
            m_hidden6_output[i] = 0
            v_hidden6_output[i] = 0
            wd_hidden6_output[i] = 0
        end

        -- Uczenie sieci
        epochs = iterations
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        learning_rate = 0.001

        for epoch = 1, epochs do
            for i = 1, #x_train do
                -- Forward pass
                hidden1_input = {}
                hidden2_input = {}
                hidden3_input = {}
                hidden4_input = {}
                hidden5_input = {}
                hidden6_input = {}
                hidden6_output = {}
                final_input = {}

                for j = 1, hidden1_size do
                    hidden1_input[j] = 0
                    for k = 1, input_size do
                        hidden1_input[j] = hidden1_input[j] + x_train[i][k] * weights_input_hidden1[k][j]
                    end
                    hidden1_input[j] = activation_input(hidden1_input[j])
                end

                for j = 1, hidden2_size do
                    hidden2_input[j] = 0
                    for k = 1, hidden1_size do
                        hidden2_input[j] = hidden2_input[j] + hidden1_input[k] * weights_hidden1_hidden2[k][j]
                    end
                    hidden2_input[j] = activation_hd(hidden2_input[j])
                end

                for j = 1, hidden3_size do
                    hidden3_input[j] = 0
                    for k = 1, hidden2_size do
                        hidden3_input[j] = hidden3_input[j] + hidden2_input[k] * weights_hidden2_hidden3[k][j]
                    end
                    hidden3_input[j] = activation_hd(hidden3_input[j])
                end

                for j = 1, hidden4_size do
                    hidden4_input[j] = 0
                    for k = 1, hidden3_size do
                        hidden4_input[j] = hidden4_input[j] + hidden3_input[k] * weights_hidden3_hidden4[k][j]
                    end
                    hidden4_input[j] = activation_hd(hidden4_input[j])
                end

                for j = 1, hidden5_size do
                    hidden5_input[j] = 0
                    for k = 1, hidden4_size do
                        hidden5_input[j] = hidden5_input[j] + hidden4_input[k] * weights_hidden4_hidden5[k][j]
                    end
                    hidden5_input[j] = activation_hd(hidden5_input[j])
                end

                for j = 1, hidden6_size do
                    hidden6_input[j] = 0
                    for k = 1, hidden5_size do
                        hidden6_input[j] = hidden6_input[j] + hidden5_input[k] * weights_hidden5_hidden6[k][j]
                    end
                    hidden6_input[j] = activation_hd(hidden6_input[j])
                end

                for j = 1, hidden6_size do
                    hidden6_output[j] = activation_hd(hidden6_input[j])
                end

                final_input[1] = 0
                for j = 1, hidden6_size do
                    final_input[1] = final_input[1] + hidden6_output[j] * weights_hidden6_output[j]
                end
                final_output = activation_output(final_input[1])

                -- Obliczenie błędu
                error = y_train[i] - final_output

                -- Backpropagation
                delta_output = error * final_output * (1 - final_output)

                delta_hidden6 = {}
                for j = 1, hidden6_size do
                    delta_hidden6[j] = delta_output * weights_hidden6_output[j] * (1 - hidden6_output[j])
                end

                delta_hidden5 = {}
                for j = 1, hidden5_size do
                    delta_hidden5[j] = 0
                    for k = 1, hidden6_size do
                        delta_hidden5[j] = delta_hidden5[j]
                            + delta_hidden6[k] * weights_hidden5_hidden6[j][k] * (1 - hidden5_input[j])
                    end
                end

                delta_hidden4 = {}
                for j = 1, hidden4_size do
                    delta_hidden4[j] = 0
                    for k = 1, hidden5_size do
                        delta_hidden4[j] = delta_hidden4[j]
                            + delta_hidden5[k] * weights_hidden4_hidden5[j][k] * (1 - hidden4_input[j])
                    end
                end

                delta_hidden3 = {}
                for j = 1, hidden3_size do
                    delta_hidden3[j] = 0
                    for k = 1, hidden4_size do
                        delta_hidden3[j] = delta_hidden3[j]
                            + delta_hidden4[k] * weights_hidden3_hidden4[j][k] * (1 - hidden3_input[j])
                    end
                end

                delta_hidden2 = {}
                for j = 1, hidden2_size do
                    delta_hidden2[j] = 0
                    for k = 1, hidden3_size do
                        delta_hidden2[j] = delta_hidden2[j]
                            + delta_hidden3[k] * weights_hidden2_hidden3[j][k] * (1 - hidden2_input[j])
                    end
                end

                delta_hidden1 = {}
                for j = 1, hidden1_size do
                    delta_hidden1[j] = 0
                    for k = 1, hidden2_size do
                        delta_hidden1[j] = delta_hidden1[j]
                            + delta_hidden2[k] * weights_hidden1_hidden2[j][k] * (1 - hidden1_input[j])
                    end
                end

                -- Aktualizacja wag zgodnie z AdamW
                for j = 1, hidden6_size do
                    m_hidden6_output[j] = beta1 * m_hidden6_output[j] + (1 - beta1) * delta_output * hidden6_output[j]
                    v_hidden6_output[j] = beta2 * v_hidden6_output[j]
                        + (1 - beta2) * (delta_output * hidden6_output[j]) ^ 2
                    m_hat = m_hidden6_output[j] / (1 - beta1 ^ epoch)
                    v_hat = v_hidden6_output[j] / (1 - beta2 ^ epoch)
                    wd_hidden6_output[j] = weight_decay * weights_hidden6_output[j]
                    weights_hidden6_output[j] = weights_hidden6_output[j]
                        + learning_rate * (m_hat / (math.sqrt(v_hat) + epsilon) - wd_hidden6_output[j])
                end

                for j = 1, hidden5_size do
                    for k = 1, hidden6_size do
                        m_hidden5_hidden6[j][k] = beta1 * m_hidden5_hidden6[j][k]
                            + (1 - beta1) * delta_hidden6[k] * hidden5_input[j]
                        v_hidden5_hidden6[j][k] = beta2 * v_hidden5_hidden6[j][k]
                            + (1 - beta2) * (delta_hidden6[k] * hidden5_input[j]) ^ 2
                        m_hat = m_hidden5_hidden6[j][k] / (1 - beta1 ^ epoch)
                        v_hat = v_hidden5_hidden6[j][k] / (1 - beta2 ^ epoch)
                        wd_hidden5_hidden6[j][k] = weight_decay * weights_hidden5_hidden6[j][k]
                        weights_hidden5_hidden6[j][k] = weights_hidden5_hidden6[j][k]
                            + learning_rate * (m_hat / (math.sqrt(v_hat) + epsilon) - wd_hidden5_hidden6[j][k])
                    end
                end

                for j = 1, hidden4_size do
                    for k = 1, hidden5_size do
                        m_hidden4_hidden5[j][k] = beta1 * m_hidden4_hidden5[j][k]
                            + (1 - beta1) * delta_hidden5[k] * hidden4_input[j]
                        v_hidden4_hidden5[j][k] = beta2 * v_hidden4_hidden5[j][k]
                            + (1 - beta2) * (delta_hidden5[k] * hidden4_input[j]) ^ 2
                        m_hat = m_hidden4_hidden5[j][k] / (1 - beta1 ^ epoch)
                        v_hat = v_hidden4_hidden5[j][k] / (1 - beta2 ^ epoch)
                        wd_hidden4_hidden5[j][k] = weight_decay * weights_hidden4_hidden5[j][k]
                        weights_hidden4_hidden5[j][k] = weights_hidden4_hidden5[j][k]
                            + learning_rate * (m_hat / (math.sqrt(v_hat) + epsilon) - wd_hidden4_hidden5[j][k])
                    end
                end

                for j = 1, hidden3_size do
                    for k = 1, hidden4_size do
                        m_hidden3_hidden4[j][k] = beta1 * m_hidden3_hidden4[j][k]
                            + (1 - beta1) * delta_hidden4[k] * hidden3_input[j]
                        v_hidden3_hidden4[j][k] = beta2 * v_hidden3_hidden4[j][k]
                            + (1 - beta2) * (delta_hidden4[k] * hidden3_input[j]) ^ 2
                        m_hat = m_hidden3_hidden4[j][k] / (1 - beta1 ^ epoch)
                        v_hat = v_hidden3_hidden4[j][k] / (1 - beta2 ^ epoch)
                        wd_hidden3_hidden4[j][k] = weight_decay * weights_hidden3_hidden4[j][k]
                        weights_hidden3_hidden4[j][k] = weights_hidden3_hidden4[j][k]
                            + learning_rate * (m_hat / (math.sqrt(v_hat) + epsilon) - wd_hidden3_hidden4[j][k])
                    end
                end

                for j = 1, hidden2_size do
                    for k = 1, hidden3_size do
                        m_hidden2_hidden3[j][k] = beta1 * m_hidden2_hidden3[j][k]
                            + (1 - beta1) * delta_hidden3[k] * hidden2_input[j]
                        v_hidden2_hidden3[j][k] = beta2 * v_hidden2_hidden3[j][k]
                            + (1 - beta2) * (delta_hidden3[k] * hidden2_input[j]) ^ 2
                        m_hat = m_hidden2_hidden3[j][k] / (1 - beta1 ^ epoch)
                        v_hat = v_hidden2_hidden3[j][k] / (1 - beta2 ^ epoch)
                        wd_hidden2_hidden3[j][k] = weight_decay * weights_hidden2_hidden3[j][k]
                        weights_hidden2_hidden3[j][k] = weights_hidden2_hidden3[j][k]
                            + learning_rate * (m_hat / (math.sqrt(v_hat) + epsilon) - wd_hidden2_hidden3[j][k])
                    end
                end

                for j = 1, hidden1_size do
                    for k = 1, hidden2_size do
                        m_hidden1_hidden2[j][k] = beta1 * m_hidden1_hidden2[j][k]
                            + (1 - beta1) * delta_hidden2[k] * hidden1_input[j]
                        v_hidden1_hidden2[j][k] = beta2 * v_hidden1_hidden2[j][k]
                            + (1 - beta2) * (delta_hidden2[k] * hidden1_input[j]) ^ 2
                        m_hat = m_hidden1_hidden2[j][k] / (1 - beta1 ^ epoch)
                        v_hat = v_hidden1_hidden2[j][k] / (1 - beta2 ^ epoch)
                        wd_hidden1_hidden2[j][k] = weight_decay * weights_hidden1_hidden2[j][k]
                        weights_hidden1_hidden2[j][k] = weights_hidden1_hidden2[j][k]
                            + learning_rate * (m_hat / (math.sqrt(v_hat) + epsilon) - wd_hidden1_hidden2[j][k])
                    end
                end

                for j = 1, input_size do
                    for k = 1, hidden1_size do
                        m_input_hidden1[j][k] = beta1 * m_input_hidden1[j][k]
                            + (1 - beta1) * delta_hidden1[k] * x_train[i][j]
                        v_input_hidden1[j][k] = beta2 * v_input_hidden1[j][k]
                            + (1 - beta2) * (delta_hidden1[k] * x_train[i][j]) ^ 2
                        m_hat = m_input_hidden1[j][k] / (1 - beta1 ^ epoch)
                        v_hat = v_input_hidden1[j][k] / (1 - beta2 ^ epoch)
                        wd_input_hidden1[j][k] = weight_decay * weights_input_hidden1[j][k]
                        weights_input_hidden1[j][k] = weights_input_hidden1[j][k]
                            + learning_rate * (m_hat / (math.sqrt(v_hat) + epsilon) - wd_input_hidden1[j][k])
                    end
                end
            end
        end

        -- Testowanie sieci
        for i = 1, #x_test do
            print("")

            hidden1_input = {}
            hidden2_input = {}
            hidden3_input = {}
            hidden4_input = {}
            hidden5_input = {}
            hidden6_input = {}
            hidden6_output = {}
            final_input = {}

            for j = 1, hidden1_size do
                hidden1_input[j] = 0
                for k = 1, input_size do
                    hidden1_input[j] = hidden1_input[j] + x_test[i][k] * weights_input_hidden1[k][j]
                end
                hidden1_input[j] = activation_input(hidden1_input[j])
            end

            for j = 1, hidden2_size do
                hidden2_input[j] = 0
                for k = 1, hidden1_size do
                    hidden2_input[j] = hidden2_input[j] + hidden1_input[k] * weights_hidden1_hidden2[k][j]
                end
                hidden2_input[j] = activation_hd(hidden2_input[j])
            end

            for j = 1, hidden3_size do
                hidden3_input[j] = 0
                for k = 1, hidden2_size do
                    hidden3_input[j] = hidden3_input[j] + hidden2_input[k] * weights_hidden2_hidden3[k][j]
                end
                hidden3_input[j] = activation_hd(hidden3_input[j])
            end

            for j = 1, hidden4_size do
                hidden4_input[j] = 0
                for k = 1, hidden3_size do
                    hidden4_input[j] = hidden4_input[j] + hidden3_input[k] * weights_hidden3_hidden4[k][j]
                end
                hidden4_input[j] = activation_hd(hidden4_input[j])
            end

            for j = 1, hidden5_size do
                hidden5_input[j] = 0
                for k = 1, hidden4_size do
                    hidden5_input[j] = hidden5_input[j] + hidden4_input[k] * weights_hidden4_hidden5[k][j]
                end
                hidden5_input[j] = activation_hd(hidden5_input[j])
            end

            for j = 1, hidden6_size do
                hidden6_input[j] = 0
                for k = 1, hidden5_size do
                    hidden6_input[j] = hidden6_input[j] + hidden5_input[k] * weights_hidden5_hidden6[k][j]
                end
                hidden6_input[j] = activation_hd(hidden6_input[j])
            end

            for j = 1, hidden6_size do
                hidden6_output[j] = activation_hd(hidden6_input[j])
            end

            final_input[1] = 0
            for j = 1, hidden6_size do
                final_input[1] = final_input[1] + hidden6_output[j] * weights_hidden6_output[j]
            end
            final_output = activation_output(final_input[1])

            print("Test input:      " .. grey, table.concat(x_test[i], " ") .. reset)
            print("Predicted output:" .. red, final_output .. reset)
        end
    end

    if hidden_layers == 7 then
        weights_input_hidden1 = {}
        weights_hidden1_hidden2 = {}
        weights_hidden2_hidden3 = {}
        weights_hidden3_hidden4 = {}
        weights_hidden4_hidden5 = {}
        weights_hidden5_hidden6 = {}
        weights_hidden6_hidden7 = {}
        weights_hidden7_output = {}

        for i = 1, input_size do
            weights_input_hidden1[i] = {}
            for j = 1, hidden1_size do
                weights_input_hidden1[i][j] = math.random()
            end
        end

        for i = 1, hidden1_size do
            weights_hidden1_hidden2[i] = {}
            for j = 1, hidden2_size do
                weights_hidden1_hidden2[i][j] = math.random()
            end
        end

        for i = 1, hidden2_size do
            weights_hidden2_hidden3[i] = {}
            for j = 1, hidden3_size do
                weights_hidden2_hidden3[i][j] = math.random()
            end
        end

        for i = 1, hidden3_size do
            weights_hidden3_hidden4[i] = {}
            for j = 1, hidden4_size do
                weights_hidden3_hidden4[i][j] = math.random()
            end
        end

        for i = 1, hidden4_size do
            weights_hidden4_hidden5[i] = {}
            for j = 1, hidden5_size do
                weights_hidden4_hidden5[i][j] = math.random()
            end
        end

        for i = 1, hidden5_size do
            weights_hidden5_hidden6[i] = {}
            for j = 1, hidden6_size do
                weights_hidden5_hidden6[i][j] = math.random()
            end
        end

        for i = 1, hidden6_size do
            weights_hidden6_hidden7[i] = {}
            for j = 1, hidden7_size do
                weights_hidden6_hidden7[i][j] = math.random()
            end
        end

        for i = 1, hidden7_size do
            weights_hidden7_output[i] = math.random()
        end

        -- Inicjalizacja zmiennych AdamW
        m_input_hidden1 = {}
        v_input_hidden1 = {}
        m_hidden1_hidden2 = {}
        v_hidden1_hidden2 = {}
        m_hidden2_hidden3 = {}
        v_hidden2_hidden3 = {}
        m_hidden3_hidden4 = {}
        v_hidden3_hidden4 = {}
        m_hidden4_hidden5 = {}
        v_hidden4_hidden5 = {}
        m_hidden5_hidden6 = {}
        v_hidden5_hidden6 = {}
        m_hidden6_hidden7 = {}
        v_hidden6_hidden7 = {}
        m_hidden7_output = {}
        v_hidden7_output = {}
        wd_input_hidden1 = {}
        wd_hidden1_hidden2 = {}
        wd_hidden2_hidden3 = {}
        wd_hidden3_hidden4 = {}
        wd_hidden4_hidden5 = {}
        wd_hidden5_hidden6 = {}
        wd_hidden6_hidden7 = {}
        wd_hidden7_output = {}

        for i = 1, input_size do
            m_input_hidden1[i] = {}
            v_input_hidden1[i] = {}
            wd_input_hidden1[i] = {}
            for j = 1, hidden1_size do
                m_input_hidden1[i][j] = 0
                v_input_hidden1[i][j] = 0
                wd_input_hidden1[i][j] = 0
            end
        end

        for i = 1, hidden1_size do
            m_hidden1_hidden2[i] = {}
            v_hidden1_hidden2[i] = {}
            wd_hidden1_hidden2[i] = {}
            for j = 1, hidden2_size do
                m_hidden1_hidden2[i][j] = 0
                v_hidden1_hidden2[i][j] = 0
                wd_hidden1_hidden2[i][j] = 0
            end
        end

        for i = 1, hidden2_size do
            m_hidden2_hidden3[i] = {}
            v_hidden2_hidden3[i] = {}
            wd_hidden2_hidden3[i] = {}
            for j = 1, hidden3_size do
                m_hidden2_hidden3[i][j] = 0
                v_hidden2_hidden3[i][j] = 0
                wd_hidden2_hidden3[i][j] = 0
            end
        end

        for i = 1, hidden3_size do
            m_hidden3_hidden4[i] = {}
            v_hidden3_hidden4[i] = {}
            wd_hidden3_hidden4[i] = {}
            for j = 1, hidden4_size do
                m_hidden3_hidden4[i][j] = 0
                v_hidden3_hidden4[i][j] = 0
                wd_hidden3_hidden4[i][j] = 0
            end
        end

        for i = 1, hidden4_size do
            m_hidden4_hidden5[i] = {}
            v_hidden4_hidden5[i] = {}
            wd_hidden4_hidden5[i] = {}
            for j = 1, hidden5_size do
                m_hidden4_hidden5[i][j] = 0
                v_hidden4_hidden5[i][j] = 0
                wd_hidden4_hidden5[i][j] = 0
            end
        end

        for i = 1, hidden5_size do
            m_hidden5_hidden6[i] = {}
            v_hidden5_hidden6[i] = {}
            wd_hidden5_hidden6[i] = {}
            for j = 1, hidden6_size do
                m_hidden5_hidden6[i][j] = 0
                v_hidden5_hidden6[i][j] = 0
                wd_hidden5_hidden6[i][j] = 0
            end
        end

        for i = 1, hidden6_size do
            m_hidden6_hidden7[i] = {}
            v_hidden6_hidden7[i] = {}
            wd_hidden6_hidden7[i] = {}
            for j = 1, hidden7_size do
                m_hidden6_hidden7[i][j] = 0
                v_hidden6_hidden7[i][j] = 0
                wd_hidden6_hidden7[i][j] = 0
            end
        end

        for i = 1, hidden7_size do
            m_hidden7_output[i] = 0
            v_hidden7_output[i] = 0
            wd_hidden7_output[i] = 0
        end

        -- Uczenie sieci
        epochs = iterations
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        learning_rate = 0.001

        for epoch = 1, epochs do
            for i = 1, #x_train do
                -- Forward pass
                hidden1_input = {}
                hidden2_input = {}
                hidden3_input = {}
                hidden4_input = {}
                hidden5_input = {}
                hidden6_input = {}
                hidden7_input = {}
                hidden7_output = {}
                final_input = {}

                for j = 1, hidden1_size do
                    hidden1_input[j] = 0
                    for k = 1, input_size do
                        hidden1_input[j] = hidden1_input[j] + x_train[i][k] * weights_input_hidden1[k][j]
                    end
                    hidden1_input[j] = activation_input(hidden1_input[j])
                end

                for j = 1, hidden2_size do
                    hidden2_input[j] = 0
                    for k = 1, hidden1_size do
                        hidden2_input[j] = hidden2_input[j] + hidden1_input[k] * weights_hidden1_hidden2[k][j]
                    end
                    hidden2_input[j] = activation_hd(hidden2_input[j])
                end

                for j = 1, hidden3_size do
                    hidden3_input[j] = 0
                    for k = 1, hidden2_size do
                        hidden3_input[j] = hidden3_input[j] + hidden2_input[k] * weights_hidden2_hidden3[k][j]
                    end
                    hidden3_input[j] = activation_hd(hidden3_input[j])
                end

                for j = 1, hidden4_size do
                    hidden4_input[j] = 0
                    for k = 1, hidden3_size do
                        hidden4_input[j] = hidden4_input[j] + hidden3_input[k] * weights_hidden3_hidden4[k][j]
                    end
                    hidden4_input[j] = activation_hd(hidden4_input[j])
                end

                for j = 1, hidden5_size do
                    hidden5_input[j] = 0
                    for k = 1, hidden4_size do
                        hidden5_input[j] = hidden5_input[j] + hidden4_input[k] * weights_hidden4_hidden5[k][j]
                    end
                    hidden5_input[j] = activation_hd(hidden5_input[j])
                end

                for j = 1, hidden6_size do
                    hidden6_input[j] = 0
                    for k = 1, hidden5_size do
                        hidden6_input[j] = hidden6_input[j] + hidden5_input[k] * weights_hidden5_hidden6[k][j]
                    end
                    hidden6_input[j] = activation_hd(hidden6_input[j])
                end

                for j = 1, hidden7_size do
                    hidden7_input[j] = 0
                    for k = 1, hidden6_size do
                        hidden7_input[j] = hidden7_input[j] + hidden6_input[k] * weights_hidden6_hidden7[k][j]
                    end
                    hidden7_input[j] = activation_hd(hidden7_input[j])
                end

                for j = 1, hidden7_size do
                    hidden7_output[j] = activation_hd(hidden7_input[j])
                end

                final_input[1] = 0
                for j = 1, hidden7_size do
                    final_input[1] = final_input[1] + hidden7_output[j] * weights_hidden7_output[j]
                end
                final_output = activation_output(final_input[1])

                -- Obliczenie błędu
                error = y_train[i] - final_output

                -- Backpropagation
                delta_output = error * final_output * (1 - final_output)

                delta_hidden7 = {}
                for j = 1, hidden7_size do
                    delta_hidden7[j] = delta_output * weights_hidden7_output[j] * (1 - hidden7_output[j])
                end

                delta_hidden6 = {}
                for j = 1, hidden6_size do
                    delta_hidden6[j] = 0
                    for k = 1, hidden7_size do
                        delta_hidden6[j] = delta_hidden6[j]
                            + delta_hidden7[k] * weights_hidden6_hidden7[j][k] * (1 - hidden6_input[j])
                    end
                end

                delta_hidden5 = {}
                for j = 1, hidden5_size do
                    delta_hidden5[j] = 0
                    for k = 1, hidden6_size do
                        delta_hidden5[j] = delta_hidden5[j]
                            + delta_hidden6[k] * weights_hidden5_hidden6[j][k] * (1 - hidden5_input[j])
                    end
                end

                delta_hidden4 = {}
                for j = 1, hidden4_size do
                    delta_hidden4[j] = 0
                    for k = 1, hidden5_size do
                        delta_hidden4[j] = delta_hidden4[j]
                            + delta_hidden5[k] * weights_hidden4_hidden5[j][k] * (1 - hidden4_input[j])
                    end
                end

                delta_hidden3 = {}
                for j = 1, hidden3_size do
                    delta_hidden3[j] = 0
                    for k = 1, hidden4_size do
                        delta_hidden3[j] = delta_hidden3[j]
                            + delta_hidden4[k] * weights_hidden3_hidden4[j][k] * (1 - hidden3_input[j])
                    end
                end

                delta_hidden2 = {}
                for j = 1, hidden2_size do
                    delta_hidden2[j] = 0
                    for k = 1, hidden3_size do
                        delta_hidden2[j] = delta_hidden2[j]
                            + delta_hidden3[k] * weights_hidden2_hidden3[j][k] * (1 - hidden2_input[j])
                    end
                end

                delta_hidden1 = {}
                for j = 1, hidden1_size do
                    delta_hidden1[j] = 0
                    for k = 1, hidden2_size do
                        delta_hidden1[j] = delta_hidden1[j]
                            + delta_hidden2[k] * weights_hidden1_hidden2[j][k] * (1 - hidden1_input[j])
                    end
                end

                -- Aktualizacja wag zgodnie z AdamW
                for j = 1, hidden7_size do
                    m_hidden7_output[j] = beta1 * m_hidden7_output[j] + (1 - beta1) * delta_output * hidden7_output[j]
                    v_hidden7_output[j] = beta2 * v_hidden7_output[j]
                        + (1 - beta2) * (delta_output * hidden7_output[j]) ^ 2
                    m_hat = m_hidden7_output[j] / (1 - beta1 ^ epoch)
                    v_hat = v_hidden7_output[j] / (1 - beta2 ^ epoch)
                    wd_hidden7_output[j] = weight_decay * weights_hidden7_output[j]
                    weights_hidden7_output[j] = weights_hidden7_output[j]
                        + learning_rate * (m_hat / (math.sqrt(v_hat) + epsilon) - wd_hidden7_output[j])
                end

                for j = 1, hidden6_size do
                    for k = 1, hidden7_size do
                        m_hidden6_hidden7[j][k] = beta1 * m_hidden6_hidden7[j][k]
                            + (1 - beta1) * delta_hidden7[k] * hidden6_input[j]
                        v_hidden6_hidden7[j][k] = beta2 * v_hidden6_hidden7[j][k]
                            + (1 - beta2) * (delta_hidden7[k] * hidden6_input[j]) ^ 2
                        m_hat = m_hidden6_hidden7[j][k] / (1 - beta1 ^ epoch)
                        v_hat = v_hidden6_hidden7[j][k] / (1 - beta2 ^ epoch)
                        wd_hidden6_hidden7[j][k] = weight_decay * weights_hidden6_hidden7[j][k]
                        weights_hidden6_hidden7[j][k] = weights_hidden6_hidden7[j][k]
                            + learning_rate * (m_hat / (math.sqrt(v_hat) + epsilon) - wd_hidden6_hidden7[j][k])
                    end
                end

                for j = 1, hidden5_size do
                    for k = 1, hidden6_size do
                        m_hidden5_hidden6[j][k] = beta1 * m_hidden5_hidden6[j][k]
                            + (1 - beta1) * delta_hidden6[k] * hidden5_input[j]
                        v_hidden5_hidden6[j][k] = beta2 * v_hidden5_hidden6[j][k]
                            + (1 - beta2) * (delta_hidden6[k] * hidden5_input[j]) ^ 2
                        m_hat = m_hidden5_hidden6[j][k] / (1 - beta1 ^ epoch)
                        v_hat = v_hidden5_hidden6[j][k] / (1 - beta2 ^ epoch)
                        wd_hidden5_hidden6[j][k] = weight_decay * weights_hidden5_hidden6[j][k]
                        weights_hidden5_hidden6[j][k] = weights_hidden5_hidden6[j][k]
                            + learning_rate * (m_hat / (math.sqrt(v_hat) + epsilon) - wd_hidden5_hidden6[j][k])
                    end
                end

                for j = 1, hidden4_size do
                    for k = 1, hidden5_size do
                        m_hidden4_hidden5[j][k] = beta1 * m_hidden4_hidden5[j][k]
                            + (1 - beta1) * delta_hidden5[k] * hidden4_input[j]
                        v_hidden4_hidden5[j][k] = beta2 * v_hidden4_hidden5[j][k]
                            + (1 - beta2) * (delta_hidden5[k] * hidden4_input[j]) ^ 2
                        m_hat = m_hidden4_hidden5[j][k] / (1 - beta1 ^ epoch)
                        v_hat = v_hidden4_hidden5[j][k] / (1 - beta2 ^ epoch)
                        wd_hidden4_hidden5[j][k] = weight_decay * weights_hidden4_hidden5[j][k]
                        weights_hidden4_hidden5[j][k] = weights_hidden4_hidden5[j][k]
                            + learning_rate * (m_hat / (math.sqrt(v_hat) + epsilon) - wd_hidden4_hidden5[j][k])
                    end
                end

                for j = 1, hidden3_size do
                    for k = 1, hidden4_size do
                        m_hidden3_hidden4[j][k] = beta1 * m_hidden3_hidden4[j][k]
                            + (1 - beta1) * delta_hidden4[k] * hidden3_input[j]
                        v_hidden3_hidden4[j][k] = beta2 * v_hidden3_hidden4[j][k]
                            + (1 - beta2) * (delta_hidden4[k] * hidden3_input[j]) ^ 2
                        m_hat = m_hidden3_hidden4[j][k] / (1 - beta1 ^ epoch)
                        v_hat = v_hidden3_hidden4[j][k] / (1 - beta2 ^ epoch)
                        wd_hidden3_hidden4[j][k] = weight_decay * weights_hidden3_hidden4[j][k]
                        weights_hidden3_hidden4[j][k] = weights_hidden3_hidden4[j][k]
                            + learning_rate * (m_hat / (math.sqrt(v_hat) + epsilon) - wd_hidden3_hidden4[j][k])
                    end
                end

                for j = 1, hidden2_size do
                    for k = 1, hidden3_size do
                        m_hidden2_hidden3[j][k] = beta1 * m_hidden2_hidden3[j][k]
                            + (1 - beta1) * delta_hidden3[k] * hidden2_input[j]
                        v_hidden2_hidden3[j][k] = beta2 * v_hidden2_hidden3[j][k]
                            + (1 - beta2) * (delta_hidden3[k] * hidden2_input[j]) ^ 2
                        m_hat = m_hidden2_hidden3[j][k] / (1 - beta1 ^ epoch)
                        v_hat = v_hidden2_hidden3[j][k] / (1 - beta2 ^ epoch)
                        wd_hidden2_hidden3[j][k] = weight_decay * weights_hidden2_hidden3[j][k]
                        weights_hidden2_hidden3[j][k] = weights_hidden2_hidden3[j][k]
                            + learning_rate * (m_hat / (math.sqrt(v_hat) + epsilon) - wd_hidden2_hidden3[j][k])
                    end
                end

                for j = 1, hidden1_size do
                    for k = 1, hidden2_size do
                        m_hidden1_hidden2[j][k] = beta1 * m_hidden1_hidden2[j][k]
                            + (1 - beta1) * delta_hidden2[k] * hidden1_input[j]
                        v_hidden1_hidden2[j][k] = beta2 * v_hidden1_hidden2[j][k]
                            + (1 - beta2) * (delta_hidden2[k] * hidden1_input[j]) ^ 2
                        m_hat = m_hidden1_hidden2[j][k] / (1 - beta1 ^ epoch)
                        v_hat = v_hidden1_hidden2[j][k] / (1 - beta2 ^ epoch)
                        wd_hidden1_hidden2[j][k] = weight_decay * weights_hidden1_hidden2[j][k]
                        weights_hidden1_hidden2[j][k] = weights_hidden1_hidden2[j][k]
                            + learning_rate * (m_hat / (math.sqrt(v_hat) + epsilon) - wd_hidden1_hidden2[j][k])
                    end
                end

                for j = 1, input_size do
                    for k = 1, hidden1_size do
                        m_input_hidden1[j][k] = beta1 * m_input_hidden1[j][k]
                            + (1 - beta1) * delta_hidden1[k] * x_train[i][j]
                        v_input_hidden1[j][k] = beta2 * v_input_hidden1[j][k]
                            + (1 - beta2) * (delta_hidden1[k] * x_train[i][j]) ^ 2
                        m_hat = m_input_hidden1[j][k] / (1 - beta1 ^ epoch)
                        v_hat = v_input_hidden1[j][k] / (1 - beta2 ^ epoch)
                        wd_input_hidden1[j][k] = weight_decay * weights_input_hidden1[j][k]
                        weights_input_hidden1[j][k] = weights_input_hidden1[j][k]
                            + learning_rate * (m_hat / (math.sqrt(v_hat) + epsilon) - wd_input_hidden1[j][k])
                    end
                end
            end
        end

        -- Testowanie sieci
        for i = 1, #x_test do
            print("")

            hidden1_input = {}
            hidden2_input = {}
            hidden3_input = {}
            hidden4_input = {}
            hidden5_input = {}
            hidden6_input = {}
            hidden7_input = {}
            hidden7_output = {}
            final_input = {}

            for j = 1, hidden1_size do
                hidden1_input[j] = 0
                for k = 1, input_size do
                    hidden1_input[j] = hidden1_input[j] + x_test[i][k] * weights_input_hidden1[k][j]
                end
                hidden1_input[j] = activation_input(hidden1_input[j])
            end

            for j = 1, hidden2_size do
                hidden2_input[j] = 0
                for k = 1, hidden1_size do
                    hidden2_input[j] = hidden2_input[j] + hidden1_input[k] * weights_hidden1_hidden2[k][j]
                end
                hidden2_input[j] = activation_hd(hidden2_input[j])
            end

            for j = 1, hidden3_size do
                hidden3_input[j] = 0
                for k = 1, hidden2_size do
                    hidden3_input[j] = hidden3_input[j] + hidden2_input[k] * weights_hidden2_hidden3[k][j]
                end
                hidden3_input[j] = activation_hd(hidden3_input[j])
            end

            for j = 1, hidden4_size do
                hidden4_input[j] = 0
                for k = 1, hidden3_size do
                    hidden4_input[j] = hidden4_input[j] + hidden3_input[k] * weights_hidden3_hidden4[k][j]
                end
                hidden4_input[j] = activation_hd(hidden4_input[j])
            end

            for j = 1, hidden5_size do
                hidden5_input[j] = 0
                for k = 1, hidden4_size do
                    hidden5_input[j] = hidden5_input[j] + hidden4_input[k] * weights_hidden4_hidden5[k][j]
                end
                hidden5_input[j] = activation_hd(hidden5_input[j])
            end

            for j = 1, hidden6_size do
                hidden6_input[j] = 0
                for k = 1, hidden5_size do
                    hidden6_input[j] = hidden6_input[j] + hidden5_input[k] * weights_hidden5_hidden6[k][j]
                end
                hidden6_input[j] = activation_hd(hidden6_input[j])
            end

            for j = 1, hidden7_size do
                hidden7_input[j] = 0
                for k = 1, hidden6_size do
                    hidden7_input[j] = hidden7_input[j] + hidden6_input[k] * weights_hidden6_hidden7[k][j]
                end
                hidden7_input[j] = activation_hd(hidden7_input[j])
            end

            for j = 1, hidden7_size do
                hidden7_output[j] = activation_hd(hidden7_input[j])
            end

            final_input[1] = 0
            for j = 1, hidden7_size do
                final_input[1] = final_input[1] + hidden7_output[j] * weights_hidden7_output[j]
            end
            final_output = activation_output(final_input[1])

            print("Test input:      " .. grey, table.concat(x_test[i], " ") .. reset)
            print("Predicted output:" .. red, final_output .. reset)
        end
    end
end
