package com.nexign.wsinteractionservice.controller;

import com.nexign.wsinteractionservice.model.OutputMessage;
import com.nexign.wsinteractionservice.service.KafkaConsumerService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/api/v0")
public class MessageController {


    private final KafkaConsumerService kafkaConsumerService;

    @Autowired
    public MessageController(KafkaConsumerService kafkaConsumerService) {
        this.kafkaConsumerService = kafkaConsumerService;
    }

    @GetMapping("/state")
    public ResponseEntity<List<OutputMessage>> getMessages(){
        return ResponseEntity.ok(kafkaConsumerService.get());
    }
}
