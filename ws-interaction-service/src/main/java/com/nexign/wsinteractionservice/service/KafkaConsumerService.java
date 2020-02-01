package com.nexign.wsinteractionservice.service;

import com.nexign.wsinteractionservice.model.Message;
import com.nexign.wsinteractionservice.model.OutputMessage;

import java.util.List;

public interface KafkaConsumerService {
    void consume(Message message);
    List<OutputMessage> get();
}
