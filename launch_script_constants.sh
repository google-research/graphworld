#!/bin/bash

function get_task_class_name()
{
  local task=$1
  case $task in
    nodeclassification) echo "NodeClassification";;
    graphregression) echo "GraphRegression";;
    linkprediction) echo "LinkPrediction";;
    noderegression) echo "NodeRegression";;
    *) echo "BAD_BASH_TASK_INPUT_${task}_";;
  esac
}
