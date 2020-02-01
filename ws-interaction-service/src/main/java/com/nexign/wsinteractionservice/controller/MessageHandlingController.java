package com.nexign.wsinteractionservice.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.stereotype.Controller;

@Controller
public class MessageHandlingController {

    private final ObjectMapper mapper = new ObjectMapper();

//    @MessageMapping("/chat")
//    @SendTo("/topic/messages")
//    public OutputMessage send(Message message){
//        return OutputMessage.builder()
//                .id(message.getId())
//                .basePhoto(message.getBasePhoto())
//                .cameraPhoto(message.getCameraPhoto())
//                .dateTime(message.getDateTime())
//                .similarity(message.getSimilarity())
//                .build();
//    }


}
