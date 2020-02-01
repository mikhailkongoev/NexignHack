package com.nexign.wsinteractionservice.service.impl;

import com.nexign.wsinteractionservice.model.Message;
import com.nexign.wsinteractionservice.model.OutputMessage;
import com.nexign.wsinteractionservice.service.KafkaConsumerService;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections.buffer.CircularFifoBuffer;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

@Slf4j
@Service
public class KafkaConsumerServiceImpl implements KafkaConsumerService {


    private final SimpMessagingTemplate simpMessagingTemplate;


    private static final int MESSAGES_MAX_NUM = 10;
    private CircularFifoBuffer circularFifoBuffer = new CircularFifoBuffer(MESSAGES_MAX_NUM);

    @Autowired
    public KafkaConsumerServiceImpl(SimpMessagingTemplate simpMessagingTemplate) {
        this.simpMessagingTemplate = simpMessagingTemplate;
    }

    @Override
    @KafkaListener(id = "ack_aton_99", topics = {"${kafka.event.topic}"}, containerFactory = "singleFactory")
    public void consume(Message message){
        log.debug("Message consumed: {}", message);
        updateBuffer(message);
        simpMessagingTemplate.convertAndSend("/topic/messages", get());
    }

    @Override
    public List<OutputMessage> get() {
        List<OutputMessage> result = new ArrayList<>();
        circularFifoBuffer.forEach(el -> result.add(map(el)));
        Collections.reverse(result);
        return result;
    }

    private  synchronized void updateBuffer(@NonNull Message message){
        circularFifoBuffer.add(message);
    }

    private OutputMessage map(@NonNull Object object){
        if (!(object instanceof Message)){
            return OutputMessage.builder().build();
        }
        Message message = (Message)object;
        return OutputMessage.builder()
                .id(String.valueOf(message.getId()))
                .basePhoto(message.getBasePhoto())
                .cameraPhoto(message.getCameraPhoto())
                .dateTime(message.getDateTime())
                .similarity(message.getSimilarity())
                .build();
    }

}
