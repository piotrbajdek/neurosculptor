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
print(grey .. "neurosculptor" .. reset .. " v0.2.0 12.09.2023")
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

if activation_input == "gelu" or activation_input == "GELU" then
    function activation_input(x)
        return 0.5
            * x
            * (
                1
                + (math.exp(2 * (math.sqrt(2 / math.pi) * (x + 0.044715 * x * x * x))) - 1)
                    / (math.exp(2 * (math.sqrt(2 / math.pi) * (x + 0.044715 * x * x * x))) + 1)
            )
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
elseif activation_input == "swish" or activation_input == "Swish" then
    function activation_input(x)
        return x / (1 + math.exp(-x))
    end
elseif activation_input == "tanh" or activation_input == "Tanh" then
    function activation_input(x)
        return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
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

if activation_hd == "gelu" or activation_hd == "GELU" then
    function activation_hd(x)
        return 0.5
            * x
            * (
                1
                + (math.exp(2 * (math.sqrt(2 / math.pi) * (x + 0.044715 * x * x * x))) - 1)
                    / (math.exp(2 * (math.sqrt(2 / math.pi) * (x + 0.044715 * x * x * x))) + 1)
            )
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
elseif activation_hd == "swish" or activation_hd == "Swish" then
    function activation_hd(x)
        return x / (1 + math.exp(-x))
    end
elseif activation_hd == "tanh" or activation_hd == "Tanh" then
    function activation_hd(x)
        return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
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

if activation_output == "gelu" or activation_output == "GELU" then
    function activation_output(x)
        return 0.5
            * x
            * (
                1
                + (math.exp(2 * (math.sqrt(2 / math.pi) * (x + 0.044715 * x * x * x))) - 1)
                    / (math.exp(2 * (math.sqrt(2 / math.pi) * (x + 0.044715 * x * x * x))) + 1)
            )
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
elseif activation_output == "swish" or activation_output == "Swish" then
    function activation_output(x)
        return x / (1 + math.exp(-x))
    end
elseif activation_output == "tanh" or activation_output == "Tanh" then
    function activation_output(x)
        return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
    end
else
    print(red .. "Invalid value in activation_output.conf" .. reset)
end

-- Otwieranie pliku do odczytu
local file = io.open("learning_rate.conf", "r")
-- Inicjalizacja zmiennej na learning_rate
local learning_rate = nil
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
if hidden_layers <= 1 or hidden_layers >= 6 then
    print(red .. "Invalid value in hidden_layers.conf" .. reset)
end

-- Otwieranie pliku do odczytu
local file = io.open("hidden_1_size.conf", "r")
-- Inicjalizacja zmiennej na iterations
local hidden1_size = nil
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

-- Otwieranie pliku do odczytu
local file = io.open("hidden_2_size.conf", "r")
-- Inicjalizacja zmiennej na iterations
local hidden2_size = nil
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

if hidden_layers == 5 then
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
if hidden_layers == 5 then
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

if hidden_layers == 5 then
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

if hidden_layers == 5 then
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
                weights_hidden2_output[j] = weights_hidden2_output[j] + learning_rate * delta_output * hidden2_output[j]
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
                weights_hidden3_output[j] = weights_hidden3_output[j] + learning_rate * delta_output * hidden3_output[j]
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
                weights_hidden4_output[j] = weights_hidden4_output[j] + learning_rate * delta_output * hidden4_output[j]
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
                weights_hidden5_output[j] = weights_hidden5_output[j] + learning_rate * delta_output * hidden5_output[j]
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
