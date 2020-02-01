package com.nexign.aggregator.aggregation;

import com.nexign.aggregator.model.AtomicEvent;
import com.nexign.aggregator.model.BusinessEvent;
import com.nexign.aggregator.producer.api.Producer;
import org.apache.commons.collections4.queue.CircularFifoQueue;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import javax.annotation.PostConstruct;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

@Service
public class Aggregator {

    @Value("${events.count}")
    private int eventsCount;

    @Value("${events.percent}")
    private double eventsPercent;

    @Value("${events.unauthorized.percent}")
    private double eventsUnauthorizedPercent;

    @Value("${similarity.threshold}")
    private double similarityThreshold;

    @Value("${events.onsuccess.delay}")
    private int onSuccessDelay;

    @Value("${events.onerror.delay}")
    private int onErrorDelay;

    private int onSuccessDelayAcc;

    private int onErrorDelayAcc;

    private Producer producer;

    private CircularFifoQueue<AtomicEvent> queue;

    private Optional<String> candidateId = Optional.empty();

    @PostConstruct
    public void init() {
        queue = new CircularFifoQueue<>(eventsCount);
    }

    @Autowired
    public Aggregator(Producer producer) {
        this.producer = producer;
    }

    public void accumulate(AtomicEvent atomicEvent) {

        if (onErrorDelayAcc > 0) {
            onErrorDelayAcc--;
            return;
        }

        if (onSuccessDelayAcc > 0) {
            onSuccessDelayAcc--;
            return;
        }

        queue.add(atomicEvent);

        if (shouldGenerateEvent()) {

            boolean authorized = isAuthorized(candidateId.get());
            BusinessEvent businessEvent = generateNewEvent(authorized);
            producer.produce(businessEvent);

            if (authorized) {
                onSuccessDelayAcc = onSuccessDelay;
            } else {
                onErrorDelayAcc = onErrorDelay;
            }

            clean();
        }
    }

    private boolean isAuthorized(String candidateId) {
        return !candidateId.equals("-1");
    }

    private boolean shouldGenerateEvent() {
        Optional<String> candidate = queue.stream()
                .collect(
                        Collectors.groupingBy(
                                AtomicEvent::getId,
                                Collectors.counting())
                ).entrySet()
                .stream()
                .filter(e -> e.getValue() >= eventsCount * eventsPercent && !e.getKey().equals("-1") ||
                        e.getValue() >= eventsCount * eventsUnauthorizedPercent && e.getKey().equals("-1"))
                .filter(e -> !e.getKey().equals("0"))
                .map(Map.Entry::getKey)
                .findAny();

        candidateId = candidate;
        return candidate.isPresent();
    }

    private BusinessEvent generateNewEvent(boolean authorized) {
        NumberFormat formatter = new DecimalFormat("#0.00");
        double similarity = queue.stream()
                .mapToDouble(event -> Double.valueOf(event.getSimilarity()))
                .average()
                .orElse(0);

        if (authorized) {
            String candidate = candidateId.get();
            AtomicEvent event = queue.stream().filter(e -> e.getId().equals(candidate)).findAny().get();
            return new BusinessEvent(candidate, event.getBasePhoto(), event.getCameraPhoto(), event.getDateTime(), formatter.format(similarity));
        } else {
            AtomicEvent event = queue.stream().filter(e -> e.getId().equals("-1")).findAny().get();
            return new BusinessEvent("-1", "", event.getCameraPhoto(), event.getDateTime(), formatter.format(similarity));
        }
    }

    private void clean() {
        candidateId = Optional.empty();
        queue.clear();
    }
}
