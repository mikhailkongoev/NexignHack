package com.nexign.aggregator.producer.impl;

import com.nexign.aggregator.model.BusinessEvent;
import com.nexign.aggregator.producer.api.Producer;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Service;

@Slf4j
@Service
public class ProducerImpl implements Producer {

    private final KafkaTemplate<Long, BusinessEvent> kafkaTemplate;

    @Value("${kafka.topic-name}")
    private String topicName;

    @Autowired
    public ProducerImpl(KafkaTemplate<Long, BusinessEvent> kafkaTemplate) {
        this.kafkaTemplate = kafkaTemplate;
    }

    @Override
    public void produce(@NonNull BusinessEvent businessEvent) {
        kafkaTemplate.send(topicName, businessEvent);
    }

}
