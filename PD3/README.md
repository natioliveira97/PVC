# PD3

Lívia Gomes Costa Fonseca - 16/0034078

Natalia Oliveira Borges - 16/0015863

## Objetivos

O objetivo desse trabalho é lidar com imagens stereo e encontrar o mapa de disparidade e profundidade entre duas imagens.

Os mapas criados foram normalizados para melhor visualização. Para desnormalizar os mapas de disparidade e profundidade deve-se aplicar a fórmula inversa da normalização:


pixel_desnomalizado = pixel_normalizado*(max-min)/255-min

## Informações

Esse trabalho foi feito em Linux Xubuntu versão 18.04 e em Linux Ubuntu versão 16.04, ambos usando Opencv versão 3.2.0. 
O código foi feito em C++.

## Relatório

O relatorio em PDF está na pasta 'relatorio'.

## Como compilar:

```
cmake src
make
```

## Como executar esse projeto

```
./pd3
```
