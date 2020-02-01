package com.nexign.aggregator.consumer;

import com.nexign.aggregator.aggregation.Aggregator;
import com.nexign.aggregator.model.AtomicEvent;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Service;

@Service
public class Consumer {
    private Aggregator aggregator;

    private int count = 0;

    @Autowired
    public Consumer(Aggregator aggregator) {
        this.aggregator = aggregator;
    }

    @KafkaListener(id = "atomic-listener", topics = "atomic", containerFactory = "singleFactory")
    public void listen(AtomicEvent atomicEvent) {
        if (count % 3 == 0) {
            for (int i = 0; i < 150; i++) {
                aggregator.accumulate(atomicEvent);
            }
        }
        aggregator.accumulate(atomicEvent);
    }
}
